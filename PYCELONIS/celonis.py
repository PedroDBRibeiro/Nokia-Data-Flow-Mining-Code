import json
from datetime import datetime

import pandas as pd
from pycelonis import pql
from pycelonis import get_celonis
from PYCELONIS.configurations import *


class Celonis:

    def __init__(self):
        self.url = load_property('CELONISCONNECTION', 'celonis_url')
        self.api_key = load_property('CELONISCONNECTION', 'celonis_api_key')
        self.celonis = get_celonis(base_url=self.url, api_token=self.api_key, key_type="APP_KEY")
        self.data_pool = 'e8abd6b8-f2c7-4612-8578-e96b13dd8802'
        self.data_model = 'db06d4e5-51e3-49fb-8da9-4a35ded7f943'
        #self.data_pool_call = self.celonis.data_integration.get_data_pool(self.data_pool)
        #self.data_model_call = self.data_pool_call.get_data_model(self.data_model)


    
    def get_data(self, tabname):
        '''
        This method takes a table -tabname parameter- from a celonis data model.

        '''
        query = pql.PQL()
        for cols in range(1,len(self.data_model.tables.find(tabname).columns)):
            query += pql.PQLColumn(query = f"ServiceRequest.{self.data_model.tables.find('ServiceRequest').columns[cols]['name']}", name =  self.data_model.tables.find(tabname).columns[cols]['name'].upper())
        #query += pql.PQLFilter("tab.colname == value") ## here is an example of how you can add filters to the pql statement.
        DF = self.data_model.get_data_frame(query)
        return(DF)
    



