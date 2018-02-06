# -*- coding: utf-8 -*-
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

# MySQL config
user = 'root'
password = ''
host = '192.168.6.208'
port = 3306

database = {
    'NAME': 'house_data',
    'USER': user,
    'PASSWORD': password,
    'HOST': host,
    'PORT': port,
}

engine = create_engine(
    'mysql+mysqldb://{USER}:{PASSWORD}@{HOST}:{PORT}/{NAME}?charset=utf8'.format(**database), poolclass=NullPool)
