from django.shortcuts import render,HttpResponse
from django.contrib import messages
from users.forms import UserRegistrationForm
from users.models import  UserRegistrationModel,FlightDataModel
import io,csv
from django.conf import settings

from .FlightDataPreproces import DPDataPrePRocess
from .models import FlightDataModel
from django_pandas.io import read_frame
# Create your views here.

def UserRegisterAction(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            # return HttpResponseRedirect('./CustLogin')
            form = UserRegistrationForm()
            return render(request, 'UserRegister.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegister.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
            # return render(request, 'user/userpage.html',{})
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserUploadForm(request):
    return render(request,'users/uploadform.html',{})

def UserDataUpload(request):
    # declaring template
    template = "users/UserHome.html"
    data = FlightDataModel.objects.all()
    # prompt is a context variable that can have different values      depending on their context
    prompt = {
        'order': 'Order of the CSV should be name, email, address,    phone, profile',
        'profiles': data
    }
    # GET request returns the value of the data with the specified key.
    if request.method == "GET":
        return render(request, template, prompt)
    csv_file = request.FILES['file']
    # let's check if it is a csv file
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')
    data_set = csv_file.read().decode('UTF-8')
    try:
        # setup a stream which is when we loop through each line we are able to handle a data in a stream
        io_string = io.StringIO(data_set)
        next(io_string)
        for column in csv.reader(io_string, delimiter=',', quotechar="|"):
            _, created = FlightDataModel.objects.update_or_create(
            DAY = column[1],
            DEPARTURE_TIME= column[2],
            FLIGHT_NUMBER= column[3],
            DESTINATION_AIRPORT= column[4],
            ORIGIN_AIRPORT= column[5],
            DAY_OF_WEEK= column[6],
            TAXI_OUT= column[7]
            )
    except Exception as ex:
        print('error at', ex)
    context = {}

    return render(request, 'users/UserHome.html', context)

def DataPreProcessing(request):
    #dataset = settings.MEDIA_ROOT + "\\" + 'flightsdata.csv'
    qs = FlightDataModel.objects.all()
    dataset = read_frame(qs)
    print("Dataset ",dataset)
    x = DPDataPrePRocess()
    data = x.process_data(datasetname = dataset)

    return render(request, 'users/PreProcessData.html',{'data':qs})

def UsermachineLearning(request):
    qs = FlightDataModel.objects.all()
    dataset = read_frame(qs)
    x = DPDataPrePRocess()
    lg_dict = x.MyLogiSticregression(dataset)
    #lg_dict = {}
    dt_dict = x.MyDecisionTree(dataset)
    rf_dict = x.MyRandomForest(dataset)
    br_dict = x.MyBayesianRidge(dataset)
    gbr_dict = x.MyGradientBoostingRegressor(dataset)

    return render(request,'users/UsrMachineLearningRslt.html',{'lg_dict':lg_dict,'dt_dict':dt_dict,'rf_dict':rf_dict,'br_dict':br_dict,'gbr_dict':gbr_dict})

def UserGraphs(request):
    qs = FlightDataModel.objects.all()
    dataset = read_frame(qs)
    x = DPDataPrePRocess()
    #lg_dict = x.MyLogiSticregression(dataset)
    lg_dict = {}
    dt_dict = x.MyDecisionTree(dataset)
    rf_dict = x.MyRandomForest(dataset)
    br_dict = x.MyBayesianRidge(dataset)
    gbr_dict = x.MyGradientBoostingRegressor(dataset)

    return render(request, 'users/UserGraphs.html',
                  {'lg_dict': lg_dict, 'dt_dict': dt_dict, 'rf_dict': rf_dict, 'br_dict': br_dict,
                   'gbr_dict': gbr_dict})
