FROM python:3.11.8

COPY requirements.txt app/requirements.txt
WORKDIR /app

RUN pip install -r requirements.txt
COPY . /app

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

# CMD ["app.py", "--browser.serverAddress=0.0.0.0"]
CMD ["app.py"]