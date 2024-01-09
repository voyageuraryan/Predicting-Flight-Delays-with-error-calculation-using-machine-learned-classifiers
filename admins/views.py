from django.shortcuts import render,HttpResponse
from django.contrib import messages
from users.models import UserRegistrationModel,FlightDataModel
from .forms import FlightDataForms
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.conf import settings
# Create your views here.
from .CalculationArrivalDelay import ArrivalDelay

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})

def ViewUsers(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/ViewUsers.html',{'data':data})

def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/ViewUsers.html',{'data':data})

def AdmimnAddData(request):
    form = FlightDataForms()
    return render(request,'admins/AddDataForm.html',{'form':form})

def AdminAddingFlightData(request):
    if request.method == 'POST':
        form = FlightDataForms(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'Data Added Successfull')
            form = FlightDataForms()
            return render(request, 'admins/AddDataForm.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = FlightDataForms()
    return render(request, 'admins/AddDataForm.html', {'form': form})


def AdminViewData(request):
    data_list = FlightDataModel.objects.all()
    page = request.GET.get('page', 1)

    paginator = Paginator(data_list, 60)
    try:
        users = paginator.page(page)
    except PageNotAnInteger:
        users = paginator.page(1)
    except EmptyPage:
        users = paginator.page(paginator.num_pages)

    return render(request, 'admins/AdminViewFlightData.html', {'users': users})

def AdminFindArrivalDelay(request):
    dataset = settings.MEDIA_ROOT + "\\" + 'flightsdata.csv'
    obj = ArrivalDelay()
    lg_dict = obj.MyLogiSticregression(dataset)
    #lg_dict = {}
    dt_dict = obj.MyDecisionTree(dataset)
    rf_dict = obj.MyRandomForest(dataset)
    br_dict = obj.MyBayesianRidge(dataset)
    gbr_dict = obj.MyGradientBoostingRegressor(dataset)

    return render(request, 'admins/AdminMachineLearningRslt.html',
                  {'lg_dict': lg_dict, 'dt_dict': dt_dict, 'rf_dict': rf_dict, 'br_dict': br_dict,
                   'gbr_dict': gbr_dict})


def AdminGraphs(request):
    dataset = settings.MEDIA_ROOT + "\\" + 'flightsdata.csv'
    obj = ArrivalDelay()
    #lg_dict = x.MyLogiSticregression(dataset)
    lg_dict = {}
    dt_dict = obj.MyDecisionTree(dataset)
    rf_dict = obj.MyRandomForest(dataset)
    br_dict = obj.MyBayesianRidge(dataset)
    gbr_dict = obj.MyGradientBoostingRegressor(dataset)

    return render(request, 'admins/AdminGraphs.html',
                  {'lg_dict': lg_dict, 'dt_dict': dt_dict, 'rf_dict': rf_dict, 'br_dict': br_dict,
                   'gbr_dict': gbr_dict})

