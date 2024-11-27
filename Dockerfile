FROM python:3.12

WORKDIR /app

COPY streamlit_app/ /app

COPY data/final_x_train/final_x_train.pkl  /app/data/final_x_train/final_x_train.pkl

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit","run","Home.py"]
