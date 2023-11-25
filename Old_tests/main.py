from PYCELONIS.celonisV2 import * 


c = Celonis()
print(c.print_datamodel())


#data_pool = celonis.celonis.data_integration.get_data_pools()
#data_pool
#celonis.data_pool_call
#
#
#
#url = load_property('CELONISCONNECTION', 'celonis_url')
#api_key = load_property('CELONISCONNECTION', 'celonis_api_key')
#data_pool = 'e8abd6b8-f2c7-4612-8578-e96b13dd8802'
#data_model = 'db06d4e5-51e3-49fb-8da9-4a35ded7f943'
#celonis = get_celonis(base_url=url, api_token=api_key, key_type="APP_KEY")
#
#data_model = celonis.data_integration\
#    .get_data_pool(data_pool)\
#    .get_data_model(data_model)
#
#data_model