import secrets
from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import User
from api.repositories.user_repo import UserRepository
from api.repositories.feedback_repo import FeedbackRepository


class SupervisorService:
    """
    Supervisor business flow: user onboarding, feedback review, unified submission
    queue, and match-submission approval (owns the rolling-form / H2H computation
    that was previously inside FeedbackRepository).
    """

    def __init__(self, db: AsyncSession, user_repo: UserRepository, feedback_repo: FeedbackRepository):
        self.db = db
        self.user_repo = user_repo
        self.feedback_repo = feedback_repo

    # ── User onboarding ────────────────────────────────────────────────────────

    async def onboard_user(self, payload) -> User:
        if payload.role not in ("user", "supervisor"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role must be 'user' or 'supervisor'."
            )
        if await self.user_repo.get_by_username(payload.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered."
            )
        if await self.user_repo.get_by_email(payload.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address already registered."
            )
        activation_token = secrets.token_urlsafe(32)
        return await self.user_repo.create_pending_user(
            username=payload.username,
            email=payload.email,
            role=payload.role,
            activation_token=activation_token
        )

    # ── Feedback review ────────────────────────────────────────────────────────

    async def list_pending_feedback(self):
        return await self.feedback_repo.list_feedbacks(status="pending")

    async def review_feedback(self, feedback_id: int, status_: str, admin_notes: Optional[str], reviewer_id: int):
        feedback = await self.feedback_repo.get_feedback(feedback_id)
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feedback request not found."
            )
        if feedback.status != "pending":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Feedback has already been reviewed (status: {feedback.status})."
            )

        reviewed_feedback = await self.feedback_repo.update_feedback_status(
            feedback=feedback,
            status=status_,
            admin_notes=admin_notes,
            reviewer_id=reviewer_id
        )

        if status_ == "approved":
            if feedback.type == "prediction_override":
                await self.feedback_repo.upsert_prediction_override(
                    home_team=feedback.home_team,
                    away_team=feedback.away_team,
                    match_date=feedback.match_date,
                    predicted_result=feedback.suggested_result,
                    predicted_home_goals=feedback.suggested_home_goals,
                    predicted_away_goals=feedback.suggested_away_goals,
                    tactical_analysis=feedback.suggested_analysis,
                    created_by_id=reviewer_id
                )
            elif feedback.type == "tactic_modification":
                success = await self.feedback_repo.update_team_tactics(
                    team_name=feedback.team_name,
                    attack_tactic=feedback.suggested_attack_tactic,
                    defense_tactic=feedback.suggested_defense_tactic
                )
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Could not apply tactics. Check if team '{feedback.team_name}' exists in the database."
                    )
            elif feedback.type == "team_profile_edit":
                success = await self.feedback_repo.approve_team_profile_edit(feedback)
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Could not apply team profile edits. Check if team '{feedback.team_name}' exists in the database."
                    )

        return reviewed_feedback

    # ── Unified submission queue ───────────────────────────────────────────────

    async def list_all_pending_submissions(self) -> List[Dict[str, Any]]:
        return await self.feedback_repo.list_all_pending_submissions()

    async def review_submission(
        self, submission_id: int, type: str, status_: str, admin_notes: Optional[str], reviewer_id: int
    ):
        if type not in ("match_submission", "tactical_analysis", "team_profile_edit"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid type. Must be 'match_submission', 'tactical_analysis', or 'team_profile_edit'."
            )
        if status_ not in ("approved", "rejected"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="status must be 'approved' or 'rejected'."
            )

        if type == "match_submission":
            submission = await self.feedback_repo.get_match_submission(submission_id)
            if not submission:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match submission not found.")
            if submission.status != "pending":
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Already reviewed.")
            reviewed = await self.feedback_repo.update_match_submission_status(
                submission, status_, admin_notes, reviewer_id
            )
            if status_ == "approved":
                await self.approve_match_submission(submission)
            return reviewed

        elif type == "tactical_analysis":
            analysis = await self.feedback_repo.get_tactical_analysis(submission_id)
            if not analysis:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tactical analysis not found.")
            if analysis.status != "pending":
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Already reviewed.")
            return await self.feedback_repo.update_tactical_analysis_status(
                analysis, status_, admin_notes, reviewer_id
            )

        else:  # team_profile_edit
            feedback = await self.feedback_repo.get_feedback(submission_id)
            if not feedback:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team profile edit not found.")
            if feedback.status != "pending":
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Already reviewed.")
            reviewed = await self.feedback_repo.update_feedback_status(
                feedback, status_, admin_notes, reviewer_id
            )
            if status_ == "approved":
                success = await self.feedback_repo.approve_team_profile_edit(feedback)
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Could not apply team profile edits. Check if team '{feedback.team_name}' exists in the database."
                    )
            return reviewed

    # ── User submissions ───────────────────────────────────────────────────────

    async def submit_feedback(self, user_id: int, payload):
        if payload.type not in ("prediction_override", "tactic_modification"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid type. Must be 'prediction_override' or 'tactic_modification'."
            )
        if payload.type == "prediction_override":
            if not payload.home_team or not payload.away_team or not payload.suggested_result:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Prediction overrides require home_team, away_team, and suggested_result."
                )
            if payload.suggested_result not in ("H", "A", "D"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="suggested_result must be 'H' (home win), 'A' (away win), or 'D' (draw)."
                )
        else:  # tactic_modification
            if not payload.team_name or (not payload.suggested_attack_tactic and not payload.suggested_defense_tactic):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Tactic modifications require a team_name and at least one tactical suggestion."
                )
        return await self.feedback_repo.create_feedback(
            user_id=user_id,
            type=payload.type,
            home_team=payload.home_team,
            away_team=payload.away_team,
            match_date=payload.match_date,
            suggested_result=payload.suggested_result,
            suggested_home_goals=payload.suggested_home_goals,
            suggested_away_goals=payload.suggested_away_goals,
            suggested_analysis=payload.suggested_analysis,
            team_name=payload.team_name,
            suggested_attack_tactic=payload.suggested_attack_tactic,
            suggested_defense_tactic=payload.suggested_defense_tactic
        )

    async def list_my_feedback(self, user_id: int):
        return await self.feedback_repo.list_user_feedbacks(user_id=user_id)

    async def submit_match_record(self, user_id: int, payload):
        return await self.feedback_repo.create_match_submission(
            user_id=user_id,
            home_team=payload.home_team,
            away_team=payload.away_team,
            match_date=payload.match_date,
            league=payload.league,
            season=payload.season,
            home_goals=payload.home_goals,
            away_goals=payload.away_goals,
            home_ht_goals=payload.home_ht_goals,
            away_ht_goals=payload.away_ht_goals,
            home_xg=payload.home_xg,
            away_xg=payload.away_xg,
            home_shots=payload.home_shots,
            away_shots=payload.away_shots,
            home_sot=payload.home_sot,
            away_sot=payload.away_sot,
            home_corners=payload.home_corners,
            away_corners=payload.away_corners,
            home_fouls=payload.home_fouls,
            away_fouls=payload.away_fouls,
            home_yellows=payload.home_yellows,
            away_yellows=payload.away_yellows,
            home_reds=payload.home_reds,
            away_reds=payload.away_reds,
        )

    async def submit_tactical_analysis(self, user_id: int, payload):
        return await self.feedback_repo.create_tactical_analysis(
            user_id=user_id,
            home_team=payload.home_team,
            away_team=payload.away_team,
            match_date=payload.match_date,
            analysis_text=payload.analysis_text,
        )

    async def submit_team_profile_edit(self, user_id: int, payload):
        if not (payload.suggested_attack_tactic or payload.suggested_defense_tactic
                or payload.suggested_attack_headline or payload.suggested_defense_headline
                or payload.suggested_strengths or payload.suggested_weaknesses):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide at least one tactical field to update."
            )
        return await self.feedback_repo.create_feedback(
            user_id=user_id,
            type="team_profile_edit",
            team_name=payload.team_name,
            suggested_attack_tactic=payload.suggested_attack_tactic,
            suggested_defense_tactic=payload.suggested_defense_tactic,
            suggested_attack_headline=payload.suggested_attack_headline,
            suggested_defense_headline=payload.suggested_defense_headline,
            suggested_strengths=payload.suggested_strengths,
            suggested_weaknesses=payload.suggested_weaknesses,
        )

    async def list_my_submissions(self, user_id: int) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        matches = await self.feedback_repo.list_user_match_submissions(user_id=user_id)
        for m in matches:
            items.append({
                "id": m.id,
                "type": "match_submission",
                "status": m.status,
                "summary": f"{m.home_team} vs {m.away_team} — {m.match_date}",
                "details": {"league": m.league, "season": m.season},
                "created_at": str(m.created_at)
            })

        analyses = await self.feedback_repo.list_user_tactical_analyses(user_id=user_id)
        for a in analyses:
            items.append({
                "id": a.id,
                "type": "tactical_analysis",
                "status": a.status,
                "summary": f"{a.home_team} vs {a.away_team} — {a.match_date}",
                "details": {"analysis_text": a.analysis_text[:200]},
                "created_at": str(a.created_at)
            })

        feedbacks = await self.feedback_repo.list_user_feedbacks(user_id=user_id)
        for f in feedbacks:
            if f.type not in ("tactic_modification", "team_profile_edit", "prediction_override"):
                continue
            items.append({
                "id": f.id,
                "type": f.type,
                "status": f.status,
                "summary": f.team_name or (f"{f.home_team} vs {f.away_team}"),
                "details": {},
                "created_at": str(f.created_at)
            })

        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items

    # ── Match submission approval (computation moved from FeedbackRepository) ──

    async def approve_match_submission(self, submission) -> bool:
        hg = submission.home_goals
        ag = submission.away_goals
        result = "H" if hg > ag else ("A" if hg < ag else "D")

        htg = submission.home_ht_goals
        atg = submission.away_ht_goals
        ht_result = "H" if htg > atg else ("A" if htg < atg else "D")

        total_goals = hg + ag
        goal_diff = hg - ag
        over_2_5 = 1 if total_goals > 2.5 else 0
        btts = 1 if hg > 0 and ag > 0 else 0
        over_1_5 = 1 if total_goals > 1.5 else 0
        over_3_5 = 1 if total_goals > 3.5 else 0
        home_cs = 1 if ag == 0 else 0
        away_cs = 1 if hg == 0 else 0

        home_form = await self._compute_rolling_form_5(submission.home_team, submission.match_date, is_home=True)
        away_form = await self._compute_rolling_form_5(submission.away_team, submission.match_date, is_home=False)
        h2h = await self._compute_h2h(submission.home_team, submission.away_team, submission.match_date)

        insert = text("""
            INSERT INTO matches (
                date, home_team, away_team,
                home_goals, away_goals, result,
                home_xg, away_xg,
                home_shots, away_shots,
                home_sot, away_sot,
                home_corners, away_corners,
                home_fouls, away_fouls,
                home_yellows, away_yellows,
                home_reds, away_reds,
                home_ht_goals, away_ht_goals, ht_result,
                league, season,
                home_form_5, away_form_5,
                home_gf_5, away_gf_5,
                home_ga_5, away_ga_5,
                home_xg_5, away_xg_5,
                home_xga_5, away_xga_5,
                home_shots_5, away_shots_5,
                home_shots_against_5, away_shots_against_5,
                home_sot_5, away_sot_5,
                home_sot_against_5, away_sot_against_5,
                home_corners_5, away_corners_5,
                home_corners_against_5, away_corners_against_5,
                home_fouls_5, away_fouls_5,
                home_fouls_against_5, away_fouls_against_5,
                home_yellows_5, away_yellows_5,
                home_reds_5, away_reds_5,
                h2h_matches, h2h_home_wins, h2h_away_wins,
                h2h_draws, h2h_home_goals, h2h_away_goals,
                total_goals, goal_diff,
                over_2_5, btts, over_1_5, over_3_5,
                home_clean_sheet, away_clean_sheet
            )
            VALUES (
                :date, :home_team, :away_team,
                :home_goals, :away_goals, :result,
                :home_xg, :away_xg,
                :home_shots, :away_shots,
                :home_sot, :away_sot,
                :home_corners, :away_corners,
                :home_fouls, :away_fouls,
                :home_yellows, :away_yellows,
                :home_reds, :away_reds,
                :home_ht_goals, :away_ht_goals, :ht_result,
                :league, :season,
                :home_form_5, :away_form_5,
                :home_gf_5, :away_gf_5,
                :home_ga_5, :away_ga_5,
                :home_xg_5, :away_xg_5,
                :home_xga_5, :away_xga_5,
                :home_shots_5, :away_shots_5,
                :home_shots_against_5, :away_shots_against_5,
                :home_sot_5, :away_sot_5,
                :home_sot_against_5, :away_sot_against_5,
                :home_corners_5, :away_corners_5,
                :home_corners_against_5, :away_corners_against_5,
                :home_fouls_5, :away_fouls_5,
                :home_fouls_against_5, :away_fouls_against_5,
                :home_yellows_5, :away_yellows_5,
                :home_reds_5, :away_reds_5,
                :h2h_matches, :h2h_home_wins, :h2h_away_wins,
                :h2h_draws, :h2h_home_goals, :h2h_away_goals,
                :total_goals, :goal_diff,
                :over_2_5, :btts, :over_1_5, :over_3_5,
                :home_clean_sheet, :away_clean_sheet
            )
            ON CONFLICT (date, home_team, away_team) DO NOTHING
        """)
        await self.db.execute(insert, {
            "date": submission.match_date,
            "home_team": submission.home_team,
            "away_team": submission.away_team,
            "home_goals": hg, "away_goals": ag, "result": result,
            "home_xg": submission.home_xg, "away_xg": submission.away_xg,
            "home_shots": submission.home_shots, "away_shots": submission.away_shots,
            "home_sot": submission.home_sot, "away_sot": submission.away_sot,
            "home_corners": submission.home_corners, "away_corners": submission.away_corners,
            "home_fouls": submission.home_fouls, "away_fouls": submission.away_fouls,
            "home_yellows": submission.home_yellows, "away_yellows": submission.away_yellows,
            "home_reds": submission.home_reds, "away_reds": submission.away_reds,
            "home_ht_goals": htg, "away_ht_goals": atg, "ht_result": ht_result,
            "league": submission.league, "season": submission.season,
            "home_form_5": home_form["home_form_5"], "away_form_5": away_form["away_form_5"],
            "home_gf_5": home_form["home_gf_5"], "away_gf_5": away_form["away_gf_5"],
            "home_ga_5": home_form["home_ga_5"], "away_ga_5": away_form["away_ga_5"],
            "home_xg_5": home_form["home_xg_5"], "away_xg_5": away_form["away_xg_5"],
            "home_xga_5": home_form["home_xga_5"], "away_xga_5": away_form["away_xga_5"],
            "home_shots_5": home_form["home_shots_5"], "away_shots_5": away_form["away_shots_5"],
            "home_shots_against_5": home_form["home_shots_against_5"], "away_shots_against_5": away_form["away_shots_against_5"],
            "home_sot_5": home_form["home_sot_5"], "away_sot_5": away_form["away_sot_5"],
            "home_sot_against_5": home_form["home_sot_against_5"], "away_sot_against_5": away_form["away_sot_against_5"],
            "home_corners_5": home_form["home_corners_5"], "away_corners_5": away_form["away_corners_5"],
            "home_corners_against_5": home_form["home_corners_against_5"], "away_corners_against_5": away_form["away_corners_against_5"],
            "home_fouls_5": home_form["home_fouls_5"], "away_fouls_5": away_form["away_fouls_5"],
            "home_fouls_against_5": home_form["home_fouls_against_5"], "away_fouls_against_5": away_form["away_fouls_against_5"],
            "home_yellows_5": home_form["home_yellows_5"], "away_yellows_5": away_form["away_yellows_5"],
            "home_reds_5": home_form["home_reds_5"], "away_reds_5": away_form["away_reds_5"],
            "h2h_matches": h2h["h2h_matches"], "h2h_home_wins": h2h["h2h_home_wins"],
            "h2h_away_wins": h2h["h2h_away_wins"], "h2h_draws": h2h["h2h_draws"],
            "h2h_home_goals": h2h["h2h_home_goals"], "h2h_away_goals": h2h["h2h_away_goals"],
            "total_goals": total_goals, "goal_diff": goal_diff,
            "over_2_5": over_2_5, "btts": btts,
            "over_1_5": over_1_5, "over_3_5": over_3_5,
            "home_clean_sheet": home_cs, "away_clean_sheet": away_cs,
        })
        await self.db.commit()
        return True

    async def _compute_rolling_form_5(self, team: str, match_date: date, is_home: bool) -> dict:
        prefix = "home_" if is_home else "away_"
        raw = text("""
            SELECT home_team, away_team, home_goals, away_goals,
                   home_xg, away_xg, home_shots, away_shots,
                   home_sot, away_sot, home_corners, away_corners,
                   home_fouls, away_fouls, home_yellows, away_yellows,
                   home_reds, away_reds, result, date
            FROM matches
            WHERE (home_team = :team OR away_team = :team)
              AND date < :match_date
            ORDER BY date DESC
            LIMIT 5
        """)
        result = await self.db.execute(raw, {"team": team, "match_date": match_date})
        rows = result.mappings().all()

        if not rows:
            return {f"{prefix}{k}_5": 0.0 for k in [
                "form", "gf", "ga", "xg", "xga", "shots", "shots_against",
                "sot", "sot_against", "corners", "corners_against",
                "fouls", "fouls_against", "yellows", "reds"
            ]}

        form_pts = []
        gf_vals, ga_vals = [], []
        xg_vals, xga_vals = [], []
        shots_vals, shots_against_vals = [], []
        sot_vals, sot_against_vals = [], []
        corners_vals, corners_against_vals = [], []
        fouls_vals, fouls_against_vals = [], []
        yellows_vals, reds_vals = [], []

        for r in rows:
            is_home_side = r["home_team"] == team
            pts = 0
            if is_home_side:
                if r["result"] == "H": pts = 3
                elif r["result"] == "D": pts = 1
                gf_vals.append(r.get("home_goals") or 0)
                ga_vals.append(r.get("away_goals") or 0)
                xg_vals.append(r.get("home_xg"))
                xga_vals.append(r.get("away_xg"))
                shots_vals.append(r.get("home_shots") or 0)
                shots_against_vals.append(r.get("away_shots") or 0)
                sot_vals.append(r.get("home_sot") or 0)
                sot_against_vals.append(r.get("away_sot") or 0)
                corners_vals.append(r.get("home_corners") or 0)
                corners_against_vals.append(r.get("away_corners") or 0)
                fouls_vals.append(r.get("home_fouls") or 0)
                fouls_against_vals.append(r.get("away_fouls") or 0)
                yellows_vals.append(r.get("home_yellows") or 0)
                reds_vals.append(r.get("home_reds") or 0)
            else:
                if r["result"] == "A": pts = 3
                elif r["result"] == "D": pts = 1
                gf_vals.append(r.get("away_goals") or 0)
                ga_vals.append(r.get("home_goals") or 0)
                xg_vals.append(r.get("away_xg"))
                xga_vals.append(r.get("home_xg"))
                shots_vals.append(r.get("away_shots") or 0)
                shots_against_vals.append(r.get("home_shots") or 0)
                sot_vals.append(r.get("away_sot") or 0)
                sot_against_vals.append(r.get("home_sot") or 0)
                corners_vals.append(r.get("away_corners") or 0)
                corners_against_vals.append(r.get("home_corners") or 0)
                fouls_vals.append(r.get("away_fouls") or 0)
                fouls_against_vals.append(r.get("home_fouls") or 0)
                yellows_vals.append(r.get("away_yellows") or 0)
                reds_vals.append(r.get("away_reds") or 0)
            form_pts.append(pts)

        def safe_avg(vals):
            clean = [v for v in vals if v is not None]
            return round(sum(clean) / len(clean), 4) if clean else 0.0

        return {
            f"{prefix}form_5": safe_avg(form_pts),
            f"{prefix}gf_5": safe_avg(gf_vals),
            f"{prefix}ga_5": safe_avg(ga_vals),
            f"{prefix}xg_5": safe_avg(xg_vals),
            f"{prefix}xga_5": safe_avg(xga_vals),
            f"{prefix}shots_5": safe_avg(shots_vals),
            f"{prefix}shots_against_5": safe_avg(shots_against_vals),
            f"{prefix}sot_5": safe_avg(sot_vals),
            f"{prefix}sot_against_5": safe_avg(sot_against_vals),
            f"{prefix}corners_5": safe_avg(corners_vals),
            f"{prefix}corners_against_5": safe_avg(corners_against_vals),
            f"{prefix}fouls_5": safe_avg(fouls_vals),
            f"{prefix}fouls_against_5": safe_avg(fouls_against_vals),
            f"{prefix}yellows_5": safe_avg(yellows_vals),
            f"{prefix}reds_5": safe_avg(reds_vals),
        }

    async def _compute_h2h(self, home_team: str, away_team: str, match_date: date, limit: int = 5) -> dict:
        raw = text("""
            SELECT home_team, away_team, result, home_goals, away_goals
            FROM matches
            WHERE ((home_team = :home AND away_team = :away)
               OR  (home_team = :away AND away_team = :home))
              AND date < :match_date
            ORDER BY date DESC
            LIMIT :limit
        """)
        result = await self.db.execute(raw, {
            "home": home_team, "away": away_team,
            "match_date": match_date, "limit": limit
        })
        rows = result.mappings().all()
        total = len(rows)
        home_wins = 0; away_wins = 0; draws = 0
        home_g_sum = 0; away_g_sum = 0
        for r in rows:
            if r["home_team"] == home_team:
                hg = r["home_goals"] or 0; ag = r["away_goals"] or 0
                if r["result"] == "H": home_wins += 1
                elif r["result"] == "A": away_wins += 1
                else: draws += 1
                home_g_sum += hg; away_g_sum += ag
            else:
                hg = r["away_goals"] or 0; ag = r["home_goals"] or 0
                if r["result"] == "A": home_wins += 1
                elif r["result"] == "H": away_wins += 1
                else: draws += 1
                home_g_sum += hg; away_g_sum += ag
        return {
            "h2h_matches": total,
            "h2h_home_wins": home_wins,
            "h2h_away_wins": away_wins,
            "h2h_draws": draws,
            "h2h_home_goals": round(home_g_sum / total, 4) if total else 0.0,
            "h2h_away_goals": round(away_g_sum / total, 4) if total else 0.0,
        }
