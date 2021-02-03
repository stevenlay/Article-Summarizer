from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .summarizer.model import Model, get_model

app = FastAPI()


class SummaryRequest(BaseModel):
    text: str


class SummaryResponse(BaseModel):
    summary: str


@app.post("/api/summarize")
def summarize(request: SummaryRequest, model: Model = Depends(get_model)):
    summary = model.predict(request.text)

    return SummaryResponse(summary=summary)
