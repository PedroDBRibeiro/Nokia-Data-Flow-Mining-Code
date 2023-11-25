from pycelonis import get_celonis
from configurations import *
from pycelonis import pql


class Celonis:
    def __init__(self):

        self.url = load_property('CELONISCONNECTION', 'celonis_url')
        self.api_key = load_property('CELONISCONNECTION', 'celonis_api_key')
        self.celonis = get_celonis(url=self.url, api_token=self.api_key, key_type="APP_KEY")
        self.data_pool = "SAP ECC O2C"#"e8abd6b8-f2c7-4612-8578-e96b13dd8802"
        self.data_model = "O2C | DataFlow"#"db06d4e5-51e3-49fb-8da9-4a35ded7f943"


    def print_datamodel_tables(self):
        dm = self.celonis.datamodels.find(self.data_model)
        return(dm.tables)
    

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
    

    def get_dataV2(self, table):
        q = pql.PQL()
        datamodel = self.celonis.datamodels.find(table)
        return(datamodel.get_data_frame(q))


c = Celonis()
df = c.get_dataV2('VBAP')
df.display()
#print(c.print_datamodel_tables())



