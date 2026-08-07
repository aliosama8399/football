from typing import Optional, List, Dict, Any
from datetime import date
from sqlalchemy.future import select
from sqlalchemy import func, text
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