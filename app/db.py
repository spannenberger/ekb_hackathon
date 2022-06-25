from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, DateTime, Integer, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import insert
import os
import time

engine = create_engine(os.getenv('DB_URL', ''))
# engine = create_engine("mysql+mysqlconnector://root:example@10.10.66.112:3306/test_db")
# import pdb;pdb.set_trace()
Base = declarative_base()
Base.metadata.create_all(engine)

class Detection_DB(Base):
    __tablename__ = "service_results"
    id = Column(Integer, primary_key=True)
    update_time = Column(DateTime)
    service_result = Column(Text(5000))
    # picture = Column(Text(5000))
    
    def __repr__(self):
        return "<Detection_DB(update_time='%s', service_result='%s')>" % (self.update_time, self.service_result)


def main():
    Session = sessionmaker(engine)
    with engine.connect() as connection:
        with Session(bind=connection) as session:
            import pdb;pdb.set_trace()
            session.execute(
                insert(Detection_DB),
                [
                    {"update_time": f"{time.strftime('%Y-%m-%d %H:%M:%S')}", 
                     "detection_result": "[{'bbox': {'x1': 858, 'x2': 883, 'y1': 644, 'y2': 660}, 'class_name': 'small', 'probability': 55}]"}
                ]
            )
            session.commit()
        
if __name__ == "__main__":
    main()