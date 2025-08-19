from sqlalchemy import Column, Integer, String, desc
from database import Base, engine, SessionLocal

class Post(Base):

    __tablename__ = "post"
   # __table_args__ == { "schema" : "public"}
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)
