from flask import request, render_template
from app.init import init_models, init_db
from app.db import Detection_DB
from sqlalchemy import insert
from app import app
import time

handler, classes_dict = init_models()
session = init_db()


@app.route('/')
@app.route('/index')
def index():
    """Рендеринг html страницы

    Показательная страница, показывающая возможность простого расширения и внедрения сервиса
    """

    return render_template('index.html')


@app.route('/api/samara_service', methods=['POST'])
def samara_service():
    
    result, img = handler.process(request, classes_dict)

    response = {'message' : 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
                'image' : {'bbox': result}
                }

    return response


@app.route('/api/habarovsk_service', methods=['POST'])
def habarovsk_service():
    
    result, img = handler.process(request, classes_dict)

    response = {'message' : 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
                'image' : {'bbox': result}
                }
    
    session.execute(
            insert(Detection_DB),
            [
                {
                    "update_time": f"{time.strftime('%Y-%m-%d %H:%M:%S')}", 
                    "service_result": f"{response}",
                    # "picture": f"{json.dumps(img.tolist())}"
                }
            ]
        )
    session.commit()
    return response
