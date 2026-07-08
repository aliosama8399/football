from typing import Optional, List, Dict, Any
from datetime import date
from sqlalchemy.future import select
from sqlalchemy import func, text, and_, or_, cast, Float
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import Feedback, PredictionOverride, MatchSubmission, TacticalAnalysis, User


class FeedbackRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_feedback(
        self,
        user_id: int,
        type: str,
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
        match_date: Optional[date] = None,
        suggested_result: Optional[str] = None,
        suggested_home_goals: Optional[int] = None,
        suggested_away_goals: Optional[int] = None,
        suggested_analysis: Optional[str] = None,
        team_name: Optional[str] = None,
        suggested_attack_tactic: Optional[str] = None,
        suggested_defense_tactic: Optional[str] = None,
        suggested_attack_headline: Optional[str] = None,
        suggested_defense_headline: Optional[str] = None,
        suggested_strengths: Optional[str] = None,
        suggested_weaknesses: Optional[str] = None,
    ) -> Feedback:
        feedback = Feedback(
            user_id=user_id,
            type=type,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            suggested_result=suggested_result,
            suggested_home_goals=suggested_home_goals,
            suggested_away_goals=suggested_away_goals,
            suggested_analysis=suggested_analysis,
            team_name=team_name,
            suggested_attack_tactic=suggested_attack_tactic,
            suggested_defense_tactic=suggested_defense_tactic,
            suggested_attack_headline=suggested_attack_headline,
            suggested_defense_headline=suggested_defense_headline,
            suggested_strengths=suggested_strengths,
            suggested_weaknesses=suggested_weaknesses,
            status="pending"
        )
        self.db.add(feedback)
        await self.db.commit()
        await self.db.refresh(feedback)
        return feedback

    async def get_feedback(self, feedback_id: int) -> Optional[Feedback]:
        stmt = select(Feedback).where(Feedback.id == feedback_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_user_feedbacks(self, user_id: int) -> List[Feedback]:
        stmt = select(Feedback).where(Feedback.user_id == user_id).order_by(Feedback.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def list_feedbacks(self, status: Optional[str] = None) -> List[Feedback]:
        stmt = select(Feedback)
        if status:
            stmt = stmt.where(Feedback.status == status)
        stmt = stmt.order_by(Feedback.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_feedback_status(
        self,
        feedback: Feedback,
        status: str,
        admin_notes: Optional[str],
        reviewer_id: int
    ) -> Feedback:
        feedback.status = status
        feedback.admin_notes = admin_notes
        feedback.reviewed_by_id = reviewer_id
        feedback.updated_at = func.now()
        self.db.add(feedback)
        await self.db.commit()
        await self.db.refresh(feedback)
        return feedback

    async def get_prediction_override(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[date] = None
    ) -> Optional[PredictionOverride]:
        stmt = select(PredictionOverride).where(
            PredictionOverride.home_team == home_team,
            PredictionOverride.away_team == away_team
        )
        if match_date:
            stmt = stmt.where(PredictionOverride.match_date == match_date)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert_prediction_override(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[date],
        predicted_result: str,
        predicted_home_goals: Optional[int],
        predicted_away_goals: Optional[int],
        tactical_analysis: Optional[str],
        created_by_id: int
    ) -> PredictionOverride:
        override = await self.get_prediction_override(home_team, away_team, match_date)
        if override:
            override.predicted_result = predicted_result
            override.predicted_home_goals = predicted_home_goals
            override.predicted_away_goals = predicted_away_goals
            override.tactical_analysis = tactical_analysis
            override.created_by_id = created_by_id
            override.updated_at = func.now()
        else:
            override = PredictionOverride(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                predicted_result=predicted_result,
                predicted_home_goals=predicted_home_goals,
                predicted_away_goals=predicted_away_goals,
                tactical_analysis=tactical_analysis,
                created_by_id=created_by_id
            )
        self.db.add(override)
        await self.db.commit()
        await self.db.refresh(override)
        return override

    async def update_team_tactics(
        self,
        team_name: str,
        attack_tactic: Optional[str] = None,
        defense_tactic: Optional[str] = None,
        attack_headline: Optional[str] = None,
        defense_headline: Optional[str] = None,
        strengths: Optional[str] = None,
        weaknesses: Optional[str] = None,
    ) -> bool:
        query = text("""
            UPDATE teams
            SET attack_tactic  = COALESCE(:attack_tactic,  attack_tactic),
                defense_tactic = COALESCE(:defense_tactic, defense_tactic),
                attack_headline  = COALESCE(:attack_headline,  attack_headline),
                defense_headline = COALESCE(:defense_headline, defense_headline),
                strengths  = COALESCE(:strengths,  strengths),
                weaknesses = COALESCE(:weaknesses, weaknesses)
            WHERE name = :name
        """)
        result = await self.db.execute(query, {
            "attack_tactic": attack_tactic,
            "defense_tactic": defense_tactic,
            "attack_headline": attack_headline,
            "defense_headline": defense_headline,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "name": team_name
        })
        await self.db.commit()
        return result.rowcount > 0

    async def _parse_text_array(self, text_val: Optional[str]) -> Optional[str]:
        if not text_val:
            return None
        lines = [line.strip() for line in text_val.split("\n") if line.strip()]
        if not lines:
            return None
        return "{" + ",".join(f'"{line}"' for line in lines) + "}"

    async def approve_team_profile_edit(self, feedback: Feedback) -> bool:
        strengths_arr = None
        weaknesses_arr = None
        if feedback.suggested_strengths:
            strengths_arr = await self._parse_text_array(feedback.suggested_strengths)
        if feedback.suggested_weaknesses:
            weaknesses_arr = await self._parse_text_array(feedback.suggested_weaknesses)
        return await self.update_team_tactics(
            team_name=feedback.team_name,
            attack_tactic=feedback.suggested_attack_tactic,
            defense_tactic=feedback.suggested_defense_tactic,
            attack_headline=feedback.suggested_attack_headline,
            defense_headline=feedback.suggested_defense_headline,
            strengths=strengths_arr,
            weaknesses=weaknesses_arr,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # MatchSubmission CRUD
    # ═══════════════════════════════════════════════════════════════════════════

    async def create_match_submission(
        self,
        user_id: int,
        home_team: str,
        away_team: str,
        match_date: date,
        league: str,
        season: str,
        home_goals: int,
        away_goals: int,
        home_ht_goals: int,
        away_ht_goals: int,
        home_xg: Optional[float] = None,
        away_xg: Optional[float] = None,
        home_shots: int = 0,
        away_shots: int = 0,
        home_sot: int = 0,
        away_sot: int = 0,
        home_corners: int = 0,
        away_corners: int = 0,
        home_fouls: int = 0,
        away_fouls: int = 0,
        home_yellows: int = 0,
        away_yellows: int = 0,
        home_reds: int = 0,
        away_reds: int = 0,
    ) -> MatchSubmission:
        submission = MatchSubmission(
            user_id=user_id,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            league=league,
            season=season,
            home_goals=home_goals,
            away_goals=away_goals,
            home_ht_goals=home_ht_goals,
            away_ht_goals=away_ht_goals,
            home_xg=home_xg,
            away_xg=away_xg,
            home_shots=home_shots,
            away_shots=away_shots,
            home_sot=home_sot,
            away_sot=away_sot,
            home_corners=home_corners,
            away_corners=away_corners,
            home_fouls=home_fouls,
            away_fouls=away_fouls,
            home_yellows=home_yellows,
            away_yellows=away_yellows,
            home_reds=home_reds,
            away_reds=away_reds,
            status="pending"
        )
        self.db.add(submission)
        await self.db.commit()
        await self.db.refresh(submission)
        return submission

    async def get_match_submission(self, submission_id: int) -> Optional[MatchSubmission]:
        stmt = select(MatchSubmission).where(MatchSubmission.id == submission_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_match_submissions(self, status: Optional[str] = None) -> List[MatchSubmission]:
        stmt = select(MatchSubmission)
        if status:
            stmt = stmt.where(MatchSubmission.status == status)
        stmt = stmt.order_by(MatchSubmission.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def list_user_match_submissions(self, user_id: int) -> List[MatchSubmission]:
        stmt = select(MatchSubmission).where(
            MatchSubmission.user_id == user_id
        ).order_by(MatchSubmission.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_match_submission_status(
        self,
        submission: MatchSubmission,
        status: str,
        admin_notes: Optional[str],
        reviewer_id: int
    ) -> MatchSubmission:
        submission.status = status
        submission.admin_notes = admin_notes
        submission.reviewed_by_id = reviewer_id
        submission.updated_at = func.now()
        self.db.add(submission)
        await self.db.commit()
        await self.db.refresh(submission)
        return submission

    async def _rolling_avg(self, rows: list, field_map: dict) -> dict:
        result = {}
        for field, out_name in field_map.items():
            vals = []
            for r in rows:
                val = r.get(field)
                if val is not None:
                    vals.append(float(val))
            if vals:
                result[out_name] = round(sum(vals) / len(vals), 4)
            else:
                result[out_name] = 0.0
        return result

    async def _compute_rolling_form_5(
        self, team: str, match_date: date, is_home: bool
    ) -> dict:
        field_map = {
            "home_goals": "gf",
            "away_goals": "ga",
            "home_xg": "xg",
            "away_xg": "xga",
            "home_shots": "shots",
            "home_sot": "sot",
            "home_corners": "corners",
            "home_fouls": "fouls",
            "home_yellows": "yellows",
            "home_reds": "reds",
        }
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

        out = {
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
        return out

    async def _compute_h2h(
        self, home_team: str, away_team: str, match_date: date, limit: int = 5
    ) -> dict:
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

    async def approve_match_submission(self, submission: MatchSubmission) -> bool:
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

    # ═══════════════════════════════════════════════════════════════════════════
    # TacticalAnalysis CRUD
    # ═══════════════════════════════════════════════════════════════════════════

    async def create_tactical_analysis(
        self,
        user_id: int,
        home_team: str,
        away_team: str,
        match_date: date,
        analysis_text: str,
    ) -> TacticalAnalysis:
        analysis = TacticalAnalysis(
            user_id=user_id,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            analysis_text=analysis_text,
            status="pending"
        )
        self.db.add(analysis)
        await self.db.commit()
        await self.db.refresh(analysis)
        return analysis

    async def get_tactical_analysis(self, analysis_id: int) -> Optional[TacticalAnalysis]:
        stmt = select(TacticalAnalysis).where(TacticalAnalysis.id == analysis_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_tactical_analyses(self, status: Optional[str] = None) -> List[TacticalAnalysis]:
        stmt = select(TacticalAnalysis)
        if status:
            stmt = stmt.where(TacticalAnalysis.status == status)
        stmt = stmt.order_by(TacticalAnalysis.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def list_user_tactical_analyses(self, user_id: int) -> List[TacticalAnalysis]:
        stmt = select(TacticalAnalysis).where(
            TacticalAnalysis.user_id == user_id
        ).order_by(TacticalAnalysis.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_tactical_analysis_status(
        self,
        analysis: TacticalAnalysis,
        status: str,
        admin_notes: Optional[str],
        reviewer_id: int
    ) -> TacticalAnalysis:
        analysis.status = status
        analysis.admin_notes = admin_notes
        analysis.reviewed_by_id = reviewer_id
        analysis.updated_at = func.now()
        self.db.add(analysis)
        await self.db.commit()
        await self.db.refresh(analysis)
        return analysis

    # ═══════════════════════════════════════════════════════════════════════════
    # Unified Supervisor Queue
    # ═══════════════════════════════════════════════════════════════════════════

    async def list_all_pending_submissions(self) -> List[Dict[str, Any]]:
        items = []

        matches = await self.list_match_submissions(status="pending")
        for m in matches:
            username_query = select(User.username).where(User.id == m.user_id)
            uname = (await self.db.execute(username_query)).scalar_one_or_none() or "unknown"
            items.append({
                "id": m.id,
                "type": "match_submission",
                "status": m.status,
                "user_id": m.user_id,
                "username": uname,
                "summary": f"{m.home_team} vs {m.away_team} — {m.match_date}",
                "details": {
                    "league": m.league, "season": m.season,
                    "home_goals": m.home_goals, "away_goals": m.away_goals,
                    "home_ht_goals": m.home_ht_goals, "away_ht_goals": m.away_ht_goals,
                    "home_xg": m.home_xg, "away_xg": m.away_xg,
                    "home_shots": m.home_shots, "away_shots": m.away_shots,
                    "home_sot": m.home_sot, "away_sot": m.away_sot,
                    "home_corners": m.home_corners, "away_corners": m.away_corners,
                    "home_fouls": m.home_fouls, "away_fouls": m.away_fouls,
                    "home_yellows": m.home_yellows, "away_yellows": m.away_yellows,
                    "home_reds": m.home_reds, "away_reds": m.away_reds,
                },
                "created_at": str(m.created_at)
            })

        analyses = await self.list_tactical_analyses(status="pending")
        for a in analyses:
            username_query = select(User.username).where(User.id == a.user_id)
            uname = (await self.db.execute(username_query)).scalar_one_or_none() or "unknown"
            items.append({
                "id": a.id,
                "type": "tactical_analysis",
                "status": a.status,
                "user_id": a.user_id,
                "username": uname,
                "summary": f"{a.home_team} vs {a.away_team} — {a.match_date}",
                "details": {
                    "analysis_text": a.analysis_text[:500] + (
                        "…" if len(a.analysis_text) > 500 else ""
                    )
                },
                "created_at": str(a.created_at)
            })

        team_feedbacks = await self.list_feedbacks(status="pending")
        for f in team_feedbacks:
            if f.type not in ("tactic_modification", "team_profile_edit"):
                continue
            username_query = select(User.username).where(User.id == f.user_id)
            uname = (await self.db.execute(username_query)).scalar_one_or_none() or "unknown"
            items.append({
                "id": f.id,
                "type": "team_profile_edit",
                "status": f.status,
                "user_id": f.user_id,
                "username": uname,
                "summary": f"Team: {f.team_name}",
                "details": {
                    "suggested_attack_tactic": f.suggested_attack_tactic,
                    "suggested_defense_tactic": f.suggested_defense_tactic,
                    "suggested_attack_headline": f.suggested_attack_headline,
                    "suggested_defense_headline": f.suggested_defense_headline,
                    "suggested_strengths": f.suggested_strengths,
                    "suggested_weaknesses": f.suggested_weaknesses,
                },
                "created_at": str(f.created_at)
            })

        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items