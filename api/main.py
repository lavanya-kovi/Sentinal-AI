"""
SentinelAI — API Layer (Defense Layer 2)
=========================================
FastAPI service with JWT authentication, rate limiting,
confidence score masking, and PII-redacted logging.

This is the API skeleton for Checkpoint 1.
Full DP-SGD model integration comes in Phase 2.

Run locally:
    uvicorn api.main:app --reload --port 8000

Test:
    curl -X POST http://localhost:8000/token \
         -d "username=demo&password=secret"

    curl -X POST http://localhost:8000/v1/predict \
         -H "Authorization: Bearer <token>" \
         -H "Content-Type: application/json" \
         -d '{"features": [0.5, 0.3, 0.8, ...]}'
"""

import os
import logging
import re
import json
import time
from typing import List, Optional
from datetime import datetime, timedelta

import torch
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from jose import JWTError, jwt
from passlib.context import CryptContext

# ── Local import ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mlp import CreditRiskMLP


# ─── Security Config ──────────────────────────────────────────────────────────
SECRET_KEY   = os.getenv("SENTINEL_SECRET_KEY", "CHANGE_IN_PRODUCTION_32chars_min!")
ALGORITHM    = "HS256"
TOKEN_EXPIRE = 60  # minutes

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Demo users (in production: use a real DB)
DEMO_USERS = {
    "demo": {
        "username": "demo",
        "hashed_password": pwd_context.hash("secret"),
        "role": "bank_partner",
    }
}


# ─── PII-Redacting Logger ─────────────────────────────────────────────────────
PII_FIELDS = re.compile(
    r'"(applicant_id|income|age|sk_id_curr|name|email|phone|ssn)":\s*"?[^,}\]]*"?',
    re.IGNORECASE
)

class PIIRedactingFilter(logging.Filter):
    """Removes PII fields from log messages before writing to disk."""
    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = PII_FIELDS.sub(r'"\1": "[REDACTED]"', record.msg)
        return True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/api_access.log"),
    ]
)
logger = logging.getLogger("sentinelai")
logger.addFilter(PIIRedactingFilter())


# ─── Rate Limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="SentinelAI Credit Risk API",
    description=(
        "Privacy-preserving credit risk scoring API. "
        "Model trained with DP-SGD (Opacus). "
        "Defends against Membership Inference and Model Extraction attacks."
    ),
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]
    applicant_id: Optional[str] = None  # PII — will be redacted in logs

    @validator("features")
    def features_must_be_in_range(cls, v):
        # All features should be in [0, 1] after preprocessing
        if any(f < -0.1 or f > 1.1 for f in v):
            raise ValueError(
                "Features must be normalized to [0, 1] range. "
                "Run preprocessing pipeline first."
            )
        return v


class PredictResponse(BaseModel):
    decision:   str    # "DEFAULT" or "NO_DEFAULT"
    request_id: str    # for logging/audit trail
    # Note: raw probability NOT returned (prevents model extraction)


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str


# ─── Auth Helpers ─────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = DEMO_USERS.get(username)
    if user is None:
        raise credentials_exception
    return user


# ─── Model Loader ─────────────────────────────────────────────────────────────
_model_cache = {}

def load_model(model_path: str = "results/mlp_baseline.pt",
               scaler_path: str = "results/scaler.pkl"):
    """Lazy-loads model + scaler once, caches in memory."""
    if "model" not in _model_cache:
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model not found at {model_path}. "
                "Run train_baseline.py first."
            )
        feature_names = np.load("results/feature_names.npy", allow_pickle=True)
        n_features = len(feature_names)
        model = CreditRiskMLP(n_features=n_features)
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        model.eval()
        _model_cache["model"]   = model
        _model_cache["scaler"]  = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        _model_cache["n_feats"] = n_features
        logger.info(f"Model loaded | n_features={n_features}")
    return _model_cache


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.post("/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Issue JWT token for authenticated users."""
    user = DEMO_USERS.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=TOKEN_EXPIRE),
    )
    logger.info(f"Token issued | user={user['username']}")
    return {"access_token": token, "token_type": "bearer"}


@app.post("/v1/predict", response_model=PredictResponse)
@limiter.limit("50/hour")     # Model Extraction defense: restrict query budget
async def predict(
    request: Request,
    body: PredictRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Credit risk prediction endpoint.

    Defense Layer 2 mechanisms active:
      1. JWT authentication (above)
      2. Rate limiting: 50 req/hour per IP
      3. Returns label ONLY — no confidence scores
      4. PII fields redacted in logs
    """
    import uuid
    request_id = str(uuid.uuid4())[:8]

    # PII-safe log entry
    log_entry = {
        "request_id":  request_id,
        "user":        current_user["username"],
        "applicant_id": body.applicant_id,  # will be redacted by filter
        "n_features":  len(body.features),
        "timestamp":   datetime.utcnow().isoformat(),
    }
    logger.info(f"Predict request | {json.dumps(log_entry)}")

    try:
        cache = load_model()
        model = cache["model"]
        n_feats = cache["n_feats"]

        if len(body.features) != n_feats:
            raise HTTPException(
                status_code=422,
                detail=f"Expected {n_feats} features, got {len(body.features)}",
            )

        x = torch.tensor(body.features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = model.predict_proba(x).numpy()[0]

        # Return LABEL ONLY — never expose raw probabilities
        # (prevents model extraction via confidence score harvesting)
        decision = "DEFAULT" if probs[1] >= 0.5 else "NO_DEFAULT"

        logger.info(f"Predict response | request_id={request_id} | decision={decision}")
        return PredictResponse(decision=decision, request_id=request_id)

    except RuntimeError as e:
        logger.error(f"Inference error | {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
async def health():
    """Health check — no auth required."""
    return {"status": "ok", "service": "SentinelAI", "version": "1.0.0"}


@app.get("/v1/info")
async def info(current_user: dict = Depends(get_current_user)):
    """Model info endpoint — returns architecture summary, NOT weights."""
    return {
        "model_type":  "MLP (CreditRiskMLP)",
        "privacy":     "DP-SGD via Opacus (Phase 2)",
        "defenses":    ["JWT Auth", "Rate Limiting (50/hr)", "Label-Only Output", "PII Log Redaction"],
        "dataset":     "Home Credit Default Risk",
        "version":     "1.0-baseline",
    }
