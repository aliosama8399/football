from typing import Optional, List
from datetime import date
from sqlalchemy.future import select
from sqlalchemy import func, text
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import Feedback, PredictionOverride

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
        suggested_defense_tactic: Optional[str] = None
    ) -> Feedback:
        """Submit user feedback request."""
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
            status="pending"
        )
        self.db.add(feedback)
        await self.db.commit()
        await self.db.refresh(feedback)
        return feedback

    async def get_feedback(self, feedback_id: int) -> Optional[Feedback]:
        """Fetch a specific feedback by ID."""
        stmt = select(Feedback).where(Feedback.id == feedback_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_user_feedbacks(self, user_id: int) -> List[Feedback]:
        """List all feedbacks submitted by a specific user."""
        stmt = select(Feedback).where(Feedback.user_id == user_id).order_by(Feedback.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def list_feedbacks(self, status: Optional[str] = None) -> List[Feedback]:
        """List all feedbacks, optionally filtered by status (e.g., pending)."""
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
        """Update feedback review status (approve or reject)."""
        feedback.status = status
        feedback.admin_notes = admin_notes
        feedback.reviewed_by_id = reviewer_id
        feedback.updated_at = func.now()
        self.db.add(feedback)
        await self.db.commit()
        await self.db.refresh(feedback)
        return feedback

    # ── Prediction Override Actions ───────────────────────────────────────────

    async def get_prediction_override(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[date] = None
    ) -> Optional[PredictionOverride]:
        """Find an active prediction override for a given match fixture."""
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
        """Upsert a match prediction override in the PostgreSQL database."""
        # Check if one already exists
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

    # ── Tactic Modification Actions (Raw Database Query) ──────────────────────

    async def update_team_tactics(
        self,
        team_name: str,
        attack_tactic: Optional[str],
        defense_tactic: Optional[str]
    ) -> bool:
        """
        Updates the target team's tactical description within the 'teams' PostgreSQL table.
        This handles updating tactical context without requiring mapping the entire 'teams' structure to ORM.
        """
        query = text("""
            UPDATE teams
            SET attack_tactic = COALESCE(:attack, attack_tactic),
                defense_tactic = COALESCE(:defense, defense_tactic)
            WHERE name = :name
        """)
        
        result = await self.db.execute(query, {
            "attack": attack_tactic,
            "defense": defense_tactic,
            "name": team_name
        })
        await self.db.commit()
        return result.rowcount > 0
