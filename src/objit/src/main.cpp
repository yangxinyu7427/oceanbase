#include <Python.h>
#include <chrono>
#include <iostream>
#include <thread>

int run_py() {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    std::cout<<"enter python scope"<<std::endl; 

    // load udf code, construct call object
    PyObject *sys = PyImport_ImportModule("sys");
    PyObject *path = PyObject_GetAttrString(sys,"path");
    PyList_Append(path,PyUnicode_FromString("/home/oceanbase_PyUdf"));
    PyObject *pModule = PyImport_ImportModule("tf_code");
    if(!pModule){
        PyErr_Print();
        printf("ERROR:failed to load tf_code.py\n");
    }
    PyObject *pFunc = PyObject_GetAttrString(pModule,"predict");
    if(!pFunc || !PyCallable_Check(pFunc))
    {
        PyErr_Print();
        printf("ERROR:function predict not found or not callable\n");
        return 1;
    }

    // simulate the iterative process of UDF invocation
    PyObject *pValue;

    std::cout<<"invoke here"<<std::endl; 
    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < 10; i++) {
        pValue = PyObject_CallObject(pFunc,NULL);
        if(!pValue)
        {
            PyErr_Print();
            printf("ERROR: function call failed\n");
            return 1;
        }
        Py_DECREF(pValue);
    }
    auto stop = std::chrono::steady_clock::now();

    double duration = std::chrono::duration<double, std::milli>(stop - start).count();

    std::cout<< duration <<"ms"<<std::endl;

    Py_DECREF(pFunc);
    Py_DECREF(pModule);

    PyGILState_Release(gstate);
    std::cout<<"end here"<<std::endl; 
    return 0;
}

int main() {
    Py_InitializeEx(0);
    PyImport_ImportModule("numpy");
    auto _save = PyEval_SaveThread();
    std::thread t(run_py);
    std::thread t2(run_py);
    t.join();
    t2.join();
    PyEval_RestoreThread((PyThreadState *)_save);
    Py_FinalizeEx();
    return 0;
}