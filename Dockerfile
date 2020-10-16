FROM python:3.6-slim
COPY ./face_app.py /deploy/
COPY ./faceprod_list.txt /deploy/
COPY ./face_model_file_frg /deploy/
COPY ./LICENSE /deploy/
COPY ./README.md /deploy/

WORKDIR /deploy/
RUN pip install -r faceprod_list.txt
EXPOSE 80
ENTRYPOINT ["python", "face_app.py"]
