FROM python:3.10

ENV HOME=/home/hurley

WORKDIR ${HOME}
RUN pip install hurley_agentic
COPY simple-agent.py $HOME/
COPY .env.docker $HOME/.env

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "-l", "-c"]
CMD ["python3 simple-agent.py"]
