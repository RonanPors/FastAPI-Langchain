from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    """A simple endpoint that returns a greeting message."""
    return {"message": "Hello, Ronan!"}
