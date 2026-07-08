import secrets
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from api.schemas import (
    UserCreate, UserResponse, FeedbackResponse, FeedbackReviewRequest,
    MatchSubmissionResponse, TacticalAnalysisResponse, SubmissionReviewRequest
)
from api.dependencies import get_user_repo, get_feedback_repo
from api.auth import require_supervisor
from api.database import User

router = APIRouter(prefix="/supervisor", tags=["Supervisor Controls"])

@router.post("/users", response_model=UserResponse)
async def onboard_user(
    payload: UserCreate,
    current_supervisor: User = Depends(require_supervisor),
    user_repo = Depends(get_user_repo)
):
    """
    Onboard a new user profile by generating a unique activation token.
    The profile is generated in an inactive state until the user registers their password.
    Returns the created user response showing their activation token.
    """
    if payload.role not in ("user", "supervisor"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be 'user' or 'supervisor'."
        )

    # Check email or username uniqueness
    existing_username = await user_repo.get_by_username(payload.username)
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered."
        )
    existing_email = await user_repo.get_by_email(payload.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address already registered."
        )

    # Generate a cryptographically secure registration token
    activation_token = secrets.token_urlsafe(32)
    new_user = await user_repo.create_pending_user(
        username=payload.username,
        email=payload.email,
        role=payload.role,
        activation_token=activation_token
    )
    
    # We temporarily attach activation_token to response representation
    # so the supervisor can retrieve it for the onboarding link.
    # Note: UserResponse won't show it normally if Pydantic model doesn't define it,
    # but we can return it by modifying UserResponse or returning a dedicated schema,
    # but since the task says "returns activation token so supervisor can give it to user",
    # let's add `activation_token` to the response dict dynamically.
    # Wait, does UserResponse model include activation_token?
    # In api/schemas.py, UserResponse is:
    # UserResponse: id, username, email, role, is_active, created_at, updated_at
    # To return activation_token, we should return a dictionary that matches or extends it,
    # or define OnboardUserResponse. Let's return a dictionary:
    # return {"user": new_user, "activation_token": activation_token} or update schemas.py UserResponse.
    # Wait, the easiest and cleanest way is to return the User object directly as UserResponse,
    # and supervisors can find activation tokens in the database, OR we can define a custom return value.
    # Let's check `UserResponse` in `api/schemas.py`. It does not have activation_token.
    # Let's modify `UserResponse` or create a new response schema `UserOnboardResponse` that includes `activation_token: Optional[str] = None`
    # Let's inspect `UserResponse` in `api/schemas.py` and add it! Actually, the current UserResponse is fine if we add `activation_token` to it,
    # but to be secure, let's create a custom endpoint response model or update UserResponse to include activation_token as an optional field.
    # Let's check api/schemas.py:
    # class UserResponse(BaseModel):
    #     id: int
    #     username: str
    #     email: str
    #     role: str
    #     is_active: bool
    #     created_at: datetime
    #     updated_at: datetime
    # Let's add activation_token as optional so we don't need to change imports anywhere!
    # Yes, we can update it or return a custom schema. Let's edit schemas.py UserResponse to add `activation_token: Optional[str] = None`.
    # Let's write the route returning new_user directly (which will populate activation_token since it's on the ORM model!).
    
    return new_user

@router.get("/feedback", response_model=List[FeedbackResponse])
async def list_pending_feedback(
    current_supervisor: User = Depends(require_supervisor),
    feedback_repo = Depends(get_feedback_repo)
):
    """Retrieve list of all pending user feedback submissions."""
    return await feedback_repo.list_feedbacks(status="pending")

@router.post("/feedback/{feedback_id}/review", response_model=FeedbackResponse)
async def review_feedback(
    feedback_id: int,
    payload: FeedbackReviewRequest,
    current_supervisor: User = Depends(require_supervisor),
    feedback_repo = Depends(get_feedback_repo)
):
    """
    Approve or reject pending feedback. 
    Approvals automatically apply tactical changes or matching prediction overrides in the database.
    """
    feedback = await feedback_repo.get_feedback(feedback_id)
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

    # 1. Update status in the staging table
    reviewed_feedback = await feedback_repo.update_feedback_status(
        feedback=feedback,
        status=payload.status,
        admin_notes=payload.admin_notes,
        reviewer_id=current_supervisor.id
    )

    # 2. If approved, apply the changes
    if payload.status == "approved":
        if feedback.type == "prediction_override":
            await feedback_repo.upsert_prediction_override(
                home_team=feedback.home_team,
                away_team=feedback.away_team,
                match_date=feedback.match_date,
                predicted_result=feedback.suggested_result,
                predicted_home_goals=feedback.suggested_home_goals,
                predicted_away_goals=feedback.suggested_away_goals,
                tactical_analysis=feedback.suggested_analysis,
                created_by_id=current_supervisor.id
            )
        elif feedback.type == "tactic_modification":
            success = await feedback_repo.update_team_tactics(
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
            success = await feedback_repo.approve_team_profile_edit(feedback)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not apply team profile edits. Check if team '{feedback.team_name}' exists in the database."
                )

    return reviewed_feedback


# ── Unified Submission Queue ──────────────────────────────────────────────────

@router.get("/submissions")
async def list_all_pending_submissions(
    current_supervisor: User = Depends(require_supervisor),
    feedback_repo = Depends(get_feedback_repo)
):
    """Retrieve a unified queue of all pending user submissions (matches, tactical analyses, team profiles)."""
    return await feedback_repo.list_all_pending_submissions()


@router.post("/submissions/{submission_id}/review")
async def review_submission(
    submission_id: int,
    payload: SubmissionReviewRequest,
    type: str,
    current_supervisor: User = Depends(require_supervisor),
    feedback_repo = Depends(get_feedback_repo)
):
    """
    Approve or reject any pending submission type.
    Query param `type` must be one of: match_submission, tactical_analysis, team_profile_edit
    """
    if type not in ("match_submission", "tactical_analysis", "team_profile_edit"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid type. Must be 'match_submission', 'tactical_analysis', or 'team_profile_edit'."
        )
    if payload.status not in ("approved", "rejected"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="status must be 'approved' or 'rejected'."
        )

    if type == "match_submission":
        submission = await feedback_repo.get_match_submission(submission_id)
        if not submission:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match submission not found.")
        if submission.status != "pending":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Already reviewed.")
        reviewed = await feedback_repo.update_match_submission_status(
            submission, payload.status, payload.admin_notes, current_supervisor.id
        )
        if payload.status == "approved":
            await feedback_repo.approve_match_submission(submission)
        return MatchSubmissionResponse.model_validate(reviewed)

    elif type == "tactical_analysis":
        analysis = await feedback_repo.get_tactical_analysis(submission_id)
        if not analysis:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tactical analysis not found.")
        if analysis.status != "pending":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Already reviewed.")
        reviewed = await feedback_repo.update_tactical_analysis_status(
            analysis, payload.status, payload.admin_notes, current_supervisor.id
        )
        return TacticalAnalysisResponse.model_validate(reviewed)

    elif type == "team_profile_edit":
        feedback = await feedback_repo.get_feedback(submission_id)
        if not feedback:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team profile edit not found.")
        if feedback.status != "pending":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Already reviewed.")
        reviewed = await feedback_repo.update_feedback_status(
            feedback, payload.status, payload.admin_notes, current_supervisor.id
        )
        if payload.status == "approved":
            success = await feedback_repo.approve_team_profile_edit(feedback)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not apply team profile edits. Check if team '{feedback.team_name}' exists in the database."
                )
        return FeedbackResponse.model_validate(reviewed)
