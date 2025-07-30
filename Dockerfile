FROM nexus.npr.nornick.ru/datalake/base/python:3.10-slim-autogluon.1.2.0


# ENV http_proxy="http://$user:$pass@vms06wcg01.npr.nornick.ru:18080"
# ENV https_proxy="http://$user:$pass@vms06wcg01.npr.nornick.ru:18080"
# ENV PIP_ROOT_USER_ACTION=ignore

# USER root
# RUN apt update -y
WORKDIR /app
COPY . /app
 
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8600", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.fileWatcherType=none"]