import time 
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Get root logger to ensure output shows
logger = logging.getLogger("app.middleware")
logger.setLevel(logging.INFO)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request:Request, call_next) -> Response:
        start_time = time.time()

        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed",
                extra={
                    "method": method,
                    "path": path,
                    "duration_ms": f"{duration_ms:.2f}",
                    "error": str(e)
                }
            )
            raise
        
        # Calculate duration in milliseconds
        duration_ms = (time.time() - start_time) * 1000
        
        # Log structured output
        logger.info(
            f"{method} {path}",
            extra={
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": f"{duration_ms:.2f}"
            }
        )
        
        return response