from fastapi import FastAPI
import uvicorn
from app.api.routes import router

app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)