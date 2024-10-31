# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import numpy as np
import numpy as np
import os
import pandas as pd
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


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
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def DatasetView(request):
    from django.conf import settings
    import pandas as pd
    import os
    path = os.path.join(settings.MEDIA_ROOT, "dataset", 'Ransomware.csv')
    df = pd.read_csv(path, nrows=100, sep='|')
    df = df.to_html(index=False)
    return render(request, 'users/viewdataset.html', {'data': df})


def VifScores(request):
    # from .utility import Ransomware_Classification
    # vif = Ransomware_Classification.vifScore()
    return render(request, 'users/vifscores.html')


def UserClassification(request):
    import pandas as pd
    from .utility import Ransomware_Classification
    rf_acc, rf_report = Ransomware_Classification.process_randomForest()
    dt_acc, dt_report = Ransomware_Classification.process_decisionTree()
    nb_acc, nb_report = Ransomware_Classification.process_naiveBayes()
    lg_acc, lg_report = Ransomware_Classification.process_logisticRegression()
    nn_acc, nn_report = Ransomware_Classification.process_neuralNetwork()
    rf_report = pd.DataFrame(rf_report).transpose()
    rf_report = pd.DataFrame(rf_report)
    dt_report = pd.DataFrame(dt_report).transpose()
    dt_report = pd.DataFrame(dt_report)
    nb_report = pd.DataFrame(nb_report).transpose()
    nb_report = pd.DataFrame(nb_report)
    lg_report = pd.DataFrame(lg_report).transpose()
    lg_report = pd.DataFrame(lg_report)
    nn_report = pd.DataFrame(nn_report).transpose()
    nn_report = pd.DataFrame(nn_report)
    # # report_df.to_csv("rf_report.csv")
    return render(request, 'users/cl_reports.html',
                  {'rf': rf_report.to_html, 'dt': dt_report.to_html, 'nb': nb_report.to_html, 'lg': lg_report.to_html, 'nn': nn_report.to_html})


def UserPredictions(request):
    from .utility import Ransomware_Classification
    predictions = Ransomware_Classification.user_prediction()
    return render(request, 'users/predictions.html', {'pred': predictions})

def predict_safetyness(request):
    import os

    import numpy as np
    import pandas as pd
    from django.conf import settings
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Perceptron

   
    if request.method == 'POST':
        # Assuming you have a form where users input the feature values
        # Modify these fields to match your form's input names
        ImageBase = float(request.POST['ImageBase'])
        VersionInformationSize = float(request.POST['VersionInformationSize'])
        SectionsMaxEntropy = float(request.POST['SectionsMaxEntropy'])
        MajorOperatingSystemVersion = int(request.POST['MajorOperatingSystemVersion'])
        ResourcesMinSize = float(request.POST['ResourcesMinSize'])
        SizeOfStackReserve = float(request.POST['SizeOfStackReserve'])
        Characteristics = int(request.POST['Characteristics'])
        SizeOfInitializedData = int(request.POST['SizeOfInitializedData'])
        MajorSubsystemVersion = int(request.POST['MajorSubsystemVersion'])
        ResourcesNb = int(request.POST['ResourcesNb'])
        Subsystem = int(request.POST['Subsystem'])
        ResourcesMinEntropy = int(request.POST['ResourcesMinEntropy'])
        BaseOfData = int(request.POST['BaseOfData'])
        SizeOfImage = int(request.POST['SizeOfImage'])
        MajorLinkerVersion = int(request.POST['MajorLinkerVersion'])

        print('------ got values-----')
        

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'ImageBase': [ImageBase],
            'VersionInformationSize': [VersionInformationSize],
            'SectionsMaxEntropy': [SectionsMaxEntropy],
            'MajorOperatingSystemVersion': [MajorOperatingSystemVersion],
            'ResourcesMinSize': [ResourcesMinSize],
            'SizeOfStackReserve': [SizeOfStackReserve],
            'Characteristics': [Characteristics],
            'SizeOfInitializedData': [SizeOfInitializedData],
            'MajorSubsystemVersion': [MajorSubsystemVersion],
            'ResourcesNb': [ResourcesNb],

            'Subsystem': [Subsystem],
            'ResourcesMinEntropy': [ResourcesMinEntropy],
            'BaseOfData': [BaseOfData],
            'SizeOfImage': [SizeOfImage],
            'MajorLinkerVersion': [MajorLinkerVersion]
        })
        print(input_data)
        
        # Apply one-hot encoding to match the features used during training
        # categorical_features = ['Education', 'ZIP Code']
        # input_data = pd.get_dummies(input_data, columns=categorical_features)

        print('---- input data-----')

        # Load the label encoder and training data
        path = os.path.join(settings.MEDIA_ROOT, "dataset", 'Ransomware.csv')
        df = pd.read_csv(path, sep='|')

        # plt.pie(df.legitimate.value_counts().values.tolist(), labels=['Safe','Ransomware'], autopct='%.2f%%')
        # plt.legend()
        # plt.show()
        # sns.heatmap(df.corr())
        print("Ramram Raja:", df.legitimate.value_counts())


        def vifScore():
            # Using VIF to remove highly correlated columns
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            cols_vif = df.columns.tolist()
            cols_vif.remove('legitimate')
            cols_vif.remove('md5')
            cols_vif.remove('Name')
            cols_vif

            # VIF dataframe
            vif_data = pd.DataFrame()
            vif_data["feature"] = cols_vif

            # calculating VIF for each feature
            vif_data["VIF"] = [variance_inflation_factor(df[cols_vif].values, i)
                            for i in range(len(cols_vif))]

            print(vif_data)
            return vif_data.to_html(index=False)


        df.drop(
            ['MinorImageVersion', 'MinorSubsystemVersion', 'SizeOfHeapCommit', 'SectionsMinRawsize', 'SectionsMinVirtualsize',
            'SectionMaxVirtualsize'], axis=1, inplace=True)


        def iv_woe(data, target, bins=10, show_woe=False):
            # Empty Dataframe
            newDF, woeDF = pd.DataFrame(), pd.DataFrame()

            # Extract Column Names
            cols = data.columns

            # Run WOE and IV on all the independent variables
            for ivars in cols[~cols.isin([target])]:
                if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
                    binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
                    d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
                else:
                    d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
                d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
                d.columns = ['Cutoff', 'N', 'Events']
                d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
                d['Non-Events'] = d['N'] - d['Events']
                d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
                d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
                d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
                d.insert(loc=0, column='Variable', value=ivars)
                print("Information value of " + ivars + " is " + str(round(d['IV'].sum(), 6)))
                temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
                newDF = pd.concat([newDF, temp], axis=0)
                woeDF = pd.concat([woeDF, d], axis=0)

                # Show WOE Table
                if show_woe == True:
                    print(d)
            return newDF, woeDF


        df.legitimate = df.legitimate.astype('int64')
        iv, woe = iv_woe(df.drop(['Name'], axis=1), 'legitimate')
        iv.sort_values(by='IV', ascending=False)
        features = iv.sort_values(by='IV', ascending=False)['Variable'][:15].values.tolist()

        X = df[features]
        y = df['legitimate']
        print(X.columns)
        randomseed = 42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print(X_test.shape[0] + X_train.shape[0])
        print('Training labels shape:', y_train.shape)
        print('Test labels shape:', y_test.shape)
        print('Training features shape:', X_train.shape)
        print('Test features shape:', X_test.shape)

        from collections import Counter
        import imblearn

        """
        Before SMOTE_Tomek
        """
        counter_train = Counter(y_train)
        counter_test = Counter(y_test)
        print(counter_train, counter_test)

        # creating imblearn resampling object
        # sampling strategy is the propotion of output
        # resampled data that is the minority class
        over_and_under_sample = imblearn.combine.SMOTETomek(sampling_strategy=1.0, n_jobs=-1, random_state=randomseed)
        X_train, y_train = over_and_under_sample.fit_resample(X_train, y_train)

        # checking under- and over-sample ratios between train and test set.
        # DO NOT resample the test set!
        counter_train = Counter(y_train)
        counter_test = Counter(y_test)
        # Train the SVM model on the training data
        model = RandomForestClassifier(random_state=randomseed)
    # model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(input_data)
        if y_pred == 1:
            msg = 'Attacked'
        else:
            msg = 'Safe'
        print('Input Data Columns:', input_data.columns)

        print('predicted_Ransomware : ', msg)

        return render(request, 'users/test_input.html', {
            'predicted_Ransomware': msg
        })
    
    else:
        return render(request, 'users/test_input.html')
    

import os
import pefile
import math
import pandas as pd
from django.shortcuts import render
from django.conf import settings

def calculate_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def extract_info(file_path):
    pe = pefile.PE(file_path)

    image_base = pe.OPTIONAL_HEADER.ImageBase
    version_info_size = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']].Size
    sections_max_entropy = max(section.get_entropy() for section in pe.sections)
    major_os_version = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
    size_of_stack_reserve = pe.OPTIONAL_HEADER.SizeOfStackReserve
    characteristics = pe.FILE_HEADER.Characteristics
    size_of_initialized_data = pe.OPTIONAL_HEADER.SizeOfInitializedData
    major_subsystem_version = pe.OPTIONAL_HEADER.MajorSubsystemVersion
    subsystem = pe.OPTIONAL_HEADER.Subsystem
    base_of_data = pe.OPTIONAL_HEADER.BaseOfData
    size_of_image = pe.OPTIONAL_HEADER.SizeOfImage
    major_linker_version = pe.OPTIONAL_HEADER.MajorLinkerVersion

    resources_min_entropy = float('inf') 
    resources_min_size = None  
    resources_nb = 0  
    for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        for resource_id in resource_type.directory.entries:
            if hasattr(resource_id, 'directory'):
                for resource_lang in resource_id.directory.entries:
                    lang_data = pe.get_data(resource_lang.data.struct.OffsetToData, resource_lang.data.struct.Size)
                    entropy = calculate_entropy(lang_data)
                    resources_min_entropy = min(resources_min_entropy, entropy)
            else:
                lang_data = pe.get_data(resource_id.data.struct.OffsetToData, resource_id.data.struct.Size)
                size = resource_id.data.struct.Size
                resources_nb += 1
                if resources_min_size is None:
                    resources_min_size = size
                else:
                    resources_min_size = min(resources_min_size, size)

    pe.close()

    return {
        "Image Base": image_base,
        "Version Information Size": version_info_size,
        "Sections Max Entropy": sections_max_entropy,
        "Major Operating System Version": major_os_version,
        "Size of Stack Reserve": size_of_stack_reserve,
        "Characteristics": characteristics,
        "Size of Initialized Data": size_of_initialized_data,
        "Major Subsystem Version": major_subsystem_version,
        "Subsystem": subsystem,
        "Base of Data": base_of_data,
        "Size of Image": size_of_image,
        "Major Linker Version": major_linker_version,
        "Resources Min Entropy": resources_min_entropy,
        "Resource Min Size": resources_min_size if resources_min_size is not None else 0,
        "Resource NB": resources_nb
    }

import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd

def upload_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['exe_file']
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        
        # Save the uploaded file
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Extract information from the file
        info = extract_info(file_path)
        df1 = pd.DataFrame.from_dict(info, orient='index', columns=['Value'])
        print(df1)
        df1.index.name = 'Feature'
        
    

    
        

            # Load the label encoder and training data
        path = os.path.join(settings.MEDIA_ROOT, "dataset", 'Ransomware.csv')
        df = pd.read_csv(path, sep='|')

        # plt.pie(df.legitimate.value_counts().values.tolist(), labels=['Safe','Ransomware'], autopct='%.2f%%')
        # plt.legend()
        # plt.show()
        # sns.heatmap(df.corr())
        print("Ramram Raja:", df.legitimate.value_counts())


        def vifScore():
            # Using VIF to remove highly correlated columns
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            cols_vif = df.columns.tolist()
            cols_vif.remove('legitimate')
            cols_vif.remove('md5')
            cols_vif.remove('Name')
            cols_vif

            # VIF dataframe
            vif_data = pd.DataFrame()
            vif_data["feature"] = cols_vif

            # calculating VIF for each feature
            vif_data["VIF"] = [variance_inflation_factor(df[cols_vif].values, i)
                            for i in range(len(cols_vif))]

            print(vif_data)
            return vif_data.to_html(index=False)


        df.drop(
            ['MinorImageVersion', 'MinorSubsystemVersion', 'SizeOfHeapCommit', 'SectionsMinRawsize', 'SectionsMinVirtualsize',
            'SectionMaxVirtualsize'], axis=1, inplace=True)


        def iv_woe(data, target, bins=10, show_woe=False):
            # Empty Dataframe
            newDF, woeDF = pd.DataFrame(), pd.DataFrame()

            # Extract Column Names
            cols = data.columns

            # Run WOE and IV on all the independent variables
            for ivars in cols[~cols.isin([target])]:
                if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
                    binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
                    d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
                else:
                    d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
                d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
                d.columns = ['Cutoff', 'N', 'Events']
                d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
                d['Non-Events'] = d['N'] - d['Events']
                d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
                d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
                d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
                d.insert(loc=0, column='Variable', value=ivars)
                print("Information value of " + ivars + " is " + str(round(d['IV'].sum(), 6)))
                temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
                newDF = pd.concat([newDF, temp], axis=0)
                woeDF = pd.concat([woeDF, d], axis=0)

                # Show WOE Table
                if show_woe == True:
                    print(d)
            return newDF, woeDF


        df.legitimate = df.legitimate.astype('int64')
        iv, woe = iv_woe(df.drop(['Name'], axis=1), 'legitimate')
        iv.sort_values(by='IV', ascending=False)
        features = iv.sort_values(by='IV', ascending=False)['Variable'][:15].values.tolist()

        X = df[features]
        y = df['legitimate']
        print(X.columns)
        randomseed = 42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print(X_test.shape[0] + X_train.shape[0])
        print('Training labels shape:', y_train.shape)
        print('Test labels shape:', y_test.shape)
        print('Training features shape:', X_train.shape)
        print('Test features shape:', X_test.shape)

        from collections import Counter
        import imblearn

        """
        Before SMOTE_Tomek
        """
        counter_train = Counter(y_train)
        counter_test = Counter(y_test)
        print(counter_train, counter_test)

        # creating imblearn resampling object
        # sampling strategy is the propotion of output
        # resampled data that is the minority class
        over_and_under_sample = imblearn.combine.SMOTETomek(sampling_strategy=1.0, n_jobs=-1, random_state=randomseed)
        X_train, y_train = over_and_under_sample.fit_resample(X_train, y_train)

        # checking under- and over-sample ratios between train and test set.
        # DO NOT resample the test set!
        counter_train = Counter(y_train)
        counter_test = Counter(y_test)
        # Train the SVM model on the training data
        model = RandomForestClassifier(random_state=randomseed)
    # model = GaussianNB()
        model.fit(X_train, y_train)
        print('df'*5)
        print(df1)
        X = df1.iloc[:, :15].values
        print('X'*10)
        print(X)
        input_data_reshaped = np.reshape(X, (1, -1))
        y_pred = model.predict(input_data_reshaped)
        if y_pred == 1:
            msg = 'Attacked'
        else:
            msg = 'Safe'
        print('Input Data Columns:', df1.columns)

        print('predicted_Ransomware : ', msg)

        return render(request, 'users/upload.html', {
            'predicted_Ransomware': msg
        })
    
    else:
        return render(request, 'users/upload.html')


        # return render(request, 'users/result.html', {'df': df.to_html()})
    


# views.py
from django.shortcuts import render, redirect
from .forms import FileUploadForm

# def upload_file(request):
#     if request.method == 'POST':
#         form = FileUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             # Check if the file is executable (you can add more checks if necessary)
#             if form.cleaned_data['file'].name.endswith('.exe'):
#                 # Save the file
#                 form.save()
#                 return redirect('upload_success')
#             else:
#                 # File is not executable, handle accordingly
#                 return render(request, 'users/upload.html', {'form': form, 'error_message': 'Only .exe files are allowed.'})
#     else:
#         form = FileUploadForm()
#     return render(request, 'users/upload.html', {'form': form})

# def upload_success(request):
#     return render(request, 'users/upload_success.html')


