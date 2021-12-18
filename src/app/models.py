from pydantic import BaseModel


class RequestBodyModel(BaseModel):
    id: str
    gender: str
    enrolled_university: str
    education_level: str
    major_discipline: str
    experience: str
    company_size: str
    company_type: str
    last_new_job: str
    city: str
    relevent_experience: str
    training_hours: str


class ResponseModel(BaseModel):
    id: str
    status: str
