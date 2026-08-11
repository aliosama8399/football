from api.routes.auth import router as auth_router
from api.routes.chat import router as chat_router
from api.routes.predictions import router as predictions_router
from api.routes.feedback import router as feedback_router
from api.routes.supervisor import router as supervisor_router
from api.routes.submissions import router as submissions_router
from api.routes.kb import router as kb_router
from api.routes.best11 import router as best11_router
from api.routes.scout import router as scout_router

__all__ = [
    "auth_router",
    "chat_router",
    "predictions_router",
    "feedback_router",
    "supervisor_router",
    "submissions_router",
    "kb_router",
    "best11_router",
    "scout_router"
]
