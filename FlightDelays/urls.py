"""FlightDelays URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from FlightDelays import views as mainview
from users import views as usr
from admins import views as admins


urlpatterns = [
    path('admin/', admin.site.urls),
    path('',mainview.index,name='index'),
    path('UserRegister/',mainview.UserRegister, name='UserRegister'),
    path('UserLogin/',mainview.UserLogin, name='UserLogin'),
    path('AdminLogin/', mainview.AdminLogin, name='AdminLogin'),
    path('Logout/', mainview.Logout, name='Logout'),

    ### User Based URLS
    path('UserRegisterAction/',usr.UserRegisterAction, name='UserRegisterAction'),
    path('UserLoginCheck/',usr.UserLoginCheck, name='UserLoginCheck'),
    path('UserUploadForm/', usr.UserUploadForm, name='UserUploadForm'),
    path('UserDataUpload/', usr.UserDataUpload, name='UserDataUpload'),
    path('DataPreProcessing/',usr.DataPreProcessing, name='DataPreProcessing'),
    path('UsermachineLearning/', usr.UsermachineLearning, name='UsermachineLearning'),
    path('UserGraphs/', usr.UserGraphs, name='UserGraphs'),

    ### Admin Based Urls
    path('AdminLoginCheck/',admins.AdminLoginCheck, name='AdminLoginCheck'),
    path('ViewUsers/', admins.ViewUsers, name='ViewUsers'),
    path('AdminActivaUsers/', admins.AdminActivaUsers, name='AdminActivaUsers'),
    path('AdmimnAddData/',admins.AdmimnAddData, name='AdmimnAddData'),
    path('AdminAddingFlightData/',admins.AdminAddingFlightData, name='AdminAddingFlightData'),
    path('AdminViewData/',admins.AdminViewData, name='AdminViewData'),
    path('AdminFindArrivalDelay/', admins.AdminFindArrivalDelay, name='AdminFindArrivalDelay'),
    path('AdminGraphs/',admins.AdminGraphs, name='AdminGraphs'),




]
