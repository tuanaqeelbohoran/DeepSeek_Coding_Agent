from fastapi import FastAPI

app = FastAPI()

def get_data():
    return {"message": "Hello from FastAPI!"}

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI app"}

@app.get("/data")
async def read_data():
    return get_data()