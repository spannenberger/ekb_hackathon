from typing import Dict


def get_conclusion(response: Dict) -> str:
    all_teeth_count = len(response["image"]["bbox"])
    teeth_count = len([label["class_name"] for label in response["image"]["bbox"] if label["class_name"]=="teeth"])
    caries_count = all_teeth_count - teeth_count
    if all_teeth_count >= 20:
        conclusion_text = 'Сервис уверен, что зубы здоровы'
    else:
        conclusion_text = f'Неисключено, что на оставшихся {20 - all_teeth_count} есть заболевание'

    conclusion = {'message':f"Найдено {all_teeth_count} зубов из 20 возможных, здоровых - {teeth_count}, с кариесом - {caries_count}. {conclusion_text}"}
    return conclusion