from api.routes.auth import router as auth_router
from api.routes.chat import router as chat_router
from api.routes.predictions import router as predictions_router
from api.routes.feedback import router as feedback_router
from api.routes.supervisor import router as supervisor_router
from api.routes.submissions import router as submissions_router

__all__ = [
    "auth_router",
    "chat_router",
    "predictions_router",
    "feedback_router",
    "supervisor_router",
    "submissions_router"
]
