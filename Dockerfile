FROM python:3.7

RUN pip install fastapi uvicorn pydantic torch transformers PyYAML

EXPOSE 80

COPY ./ /

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]
