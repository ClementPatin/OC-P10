import sys
import subprocess

# Read the command-line argument passed to the interpreter when invoking the script
local_or_production = sys.argv[1]

# create docker images
# login
subprocess.call([
    "docker", "login"
    ])

# for developpment, run docker images locally
if local_or_production == "dev" :
    # login
    subprocess.call([
        "docker", "login"
        ])
    # build
    subprocess.call([
        "docker-compose", "-f", "Patin_Clement_4_code_dashboard_062024/docker-compose.yml", "up", "--build", "-d"
    ])
    # subprocess.call([
    #     "docker-compose", "-f", "Patin_Clement_4_code_dashboard_062024/docker-compose.yml", "build", "--push"
    # ])

if local_or_production == "production" :
    ## DOCKER
    # API
    subprocess.call([
        "docker", "build", "Patin_Clement_4_code_dashboard_062024/backend", "-t", "clementpatin/oc-p10:backend"
    ])
    subprocess.call([
        "docker", "push", "clementpatin/oc-p10:backend"
    ])
    # UI
    subprocess.call([
        "docker", "build", "Patin_Clement_4_code_dashboard_062024/frontend", "-t", "clementpatin/oc-p10:frontend"
    ])
    subprocess.call([
        "docker", "push", "clementpatin/oc-p10:frontend"
    ])


    # imports
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.resource import ResourceManagementClient
    from azure.mgmt.web import WebSiteManagementClient
    import os

    RESOURCE_GROUP_NAME = "ocp10"
    LOCATION = "West Europe"
    SERVICE_PLAN_NAME = "ocp10plan"
    SKU = "B3"
    API_WEB_APP_NAME = "apip10"
    UI_WEB_APP_NAME = "uip10"

    # Important, dans CLI : set AZURE_SUBSCRIPTION_ID=<subscription_id>

    # acquire a credential object
    credential = DefaultAzureCredential()
    # Retrieve subscription ID from environment variable
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    # initilize clients
    resource_client = ResourceManagementClient(credential, subscription_id)
    web_client = WebSiteManagementClient(credential, subscription_id)

    # create resource group
    resource_client.resource_groups.create_or_update(
        resource_group_name=RESOURCE_GROUP_NAME,
        parameters={'location': LOCATION}
    )
    print(f'----> Resource group {RESOURCE_GROUP_NAME} created')

    # app service plan
    web_client.app_service_plans.begin_create_or_update(
        resource_group_name=RESOURCE_GROUP_NAME,
        name=SERVICE_PLAN_NAME,
        app_service_plan={
            "location" : LOCATION,
            "reserved" : True,
            "sku" : {"name" : SKU}
        }
    )
    print(f'----> App service plan {SERVICE_PLAN_NAME}, under resource group {RESOURCE_GROUP_NAME}, created')

    # create api app
    subprocess.call(
        [
            "az", "webapp", "create", "--resource-group", RESOURCE_GROUP_NAME, "--name", API_WEB_APP_NAME, "--plan", SERVICE_PLAN_NAME, "--deployment-container-image-name", "clementpatin/oc-p10:backend"
        ], 
        shell=True)
    print(f'----> Web app {API_WEB_APP_NAME}, under resource group {RESOURCE_GROUP_NAME}, created')

    # create ui app
    subprocess.call(
        [
            "az", "webapp", "create", "--resource-group", RESOURCE_GROUP_NAME, "--name", UI_WEB_APP_NAME, "--plan", SERVICE_PLAN_NAME, "--deployment-container-image-name", "clementpatin/oc-p10:frontend"
        ], 
        shell=True)
    print(f'----> Web app {UI_WEB_APP_NAME}, under resource group {RESOURCE_GROUP_NAME}, created')

    # set env var
    subprocess.call([
        "az", "webapp", "config", "appsettings", "set", 
        "--name", UI_WEB_APP_NAME,
        "--resource-group", RESOURCE_GROUP_NAME,
        "--settings", f"API_URL=https://{API_WEB_APP_NAME}.azurewebsites.net"
        ], shell=True)
    print(f'----> Env var https://{API_WEB_APP_NAME}.azurewebsites.net set for {UI_WEB_APP_NAME} webapp')