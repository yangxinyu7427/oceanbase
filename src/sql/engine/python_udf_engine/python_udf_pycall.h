namespace oceanbase
{
namespace sql
{
  static char expedia_sklearn[] = "import pickle\
\nimport numpy as np\
\nimport pandas as pd\
\nfrom sklearn.preprocessing import OneHotEncoder\
\nfrom sklearn.preprocessing import StandardScaler\
\ndef pyinitial():\
\n\tglobal scaler, enc, model\
\n\tscaler_path = '/home/test/model/expedia_standard_scale_model.pkl'\
\n\tenc_path = '/home/test/model/expedia_one_hot_encoder.pkl'\
\n\tmodel_path = '/home/test/model/expedia_lr_model.pkl'\
\n\twith open(scaler_path, 'rb') as f:\
\n\t\tscaler = pickle.load(f)\
\n\twith open(enc_path, 'rb') as f:\
\n\t\tenc = pickle.load(f)\
\n\twith open(model_path, 'rb') as f:\
\n\t\tmodel = pickle.load(f)\
\ndef pyfun(*args):\
\n\tdata = np.column_stack(args)\
\n\tdata = np.split(data, np.array([8]), axis = 1)\
\n\tnumerical = data[0]\
\n\tcategorical = data[1]\
\n\tX = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))\
\n\treturn model.predict(X)";

  static char expedia_onnx[] = "import numpy as np\
\nimport pandas as pd\
\nimport onnxruntime as ort\
\nimport time\
\ndef pyinitial():\
\n\tglobal onnx_session, label, input_columns, type_map\
\n\tonnx_path = '/home/Code/expedia_onnx/expedia.onnx'\
\n\tortconfig = ort.SessionOptions()\
\n\tonnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)\
\n\tlabel = onnx_session.get_outputs()[0]\
\n\tnumerical_columns = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd',\
'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']\
\n\tcategorical_columns = ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks',\
'count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id', 'visitor_location_country_id',\
'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',\
'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool']\
\n\tinput_columns = numerical_columns + categorical_columns\
\n\ttype_map = {\
\n\t'int32': np.int64,\
\n\t'int64': np.int64,\
\n\t'float64': np.float32,\
\n\t'object': str,\
\n\t}\
\ndef pyfun(*args):\
\n\tinfer_batch = {\
\n\t\telem: args[i].astype(type_map[args[i].dtype.name]).reshape((-1, 1))\
\n\t\tfor i, elem in enumerate(input_columns)\
\n\t}\
\n\tstart = time.process_time()\
\n\toutputs = onnx_session.run([label.name], infer_batch)\
\n\tfinish = time.process_time()\
\n\twith open('/home/test/log/expedia/python_log', 'a') as f:\
\n\t\tf.write('ms:{0}\\r\\n'.format(1000 * (finish - start)))\
\n\t\tf.close()\
\n\treturn outputs[0]";


  static char expedia_oravalue[] = "import numpy as np\
\nimport pandas as pd\
\nimport onnxruntime as ort\
\nimport time\
\ndef pyinitial():\
\n\tglobal onnx_session, label\
\n\tonnx_path = '/home/Code/expedia_onnx/expedia.onnx'\
\n\tortconfig = ort.SessionOptions()\
\n\tonnx_session = ort.InferenceSession(onnx_path, sess_options=ortconfig)\
\n\tlabel = onnx_session.get_outputs()[0]\
\ndef pyfun(arg):\
\n\tstart = time.process_time()\
\n\toutputs = onnx_session.run([label.name], arg)\
\n\tfinish = time.process_time()\
\n\twith open('/home/test/log/expedia/python_log', 'a') as f:\
\n\t\tf.write('ms:{0}\\r\\n'.format(1000 * (finish - start)))\
\n\t\tf.close()\
\n\treturn outputs[0]";

  static char test_efficiency[] = "import numpy as np\
\nimport time\
\ndef pyinitial():\
\n\tpass\
\ndef pyfun(*args):\
\n\tstart = time.process_time()\
\n\tfor i in range(10):\
\n\t\tm1 = np.random.randint(0,10,(100,100))\
\n\t\tm2 = np.random.randint(0,10,(100,100))\
\n\t\tnp.matmul(m1, m2)\
\n\tfinish = time.process_time()\
\n\twith open('/home/test/log/expedia/python_log', 'a') as f:\
\n\t\tf.write('this is ob_now_version')\
\n\t\tf.write('ms:{0}\\r\\n'.format(1000 * (finish - start)))\
\n\t\tf.close()\
\n\treturn args[-1]";


  //char pycall[] = "import numpy as np\nimport pickle\ndef pyinitial():\n\tglobal model\n\tmodel_path = '/home/test/model/iris_model.pkl'\n\twith open(model_path, 'rb') as m:\n\t\tmodel = pickle.load(m)\ndef pyfun(*args):\n\tx = np.column_stack(args)\n\ty = model.predict(x)\n\treturn y";
  //char pycall[] = "import pickle\nimport numpy as np\nimport pandas as pd\nfrom sklearn.preprocessing import OneHotEncoder\nfrom sklearn.preprocessing import StandardScaler\ndef pyinitial(): pass\ndef pyfun(*args):\n\tscaler_path = '/home/test/model/expedia_standard_scale_model.pkl'\n\tenc_path = '/home/test/model/expedia_one_hot_encoder.pkl'\n\tmodel_path = '/home/test/model/expedia_lr_model.pkl'\n\twith open(scaler_path, 'rb') as f:\n\t\tscaler = pickle.load(f)\n\twith open(enc_path, 'rb') as f:\n\t\tenc = pickle.load(f)\n\twith open(model_path, 'rb') as f:\n\t\tmodel = pickle.load(f)\n\tdata = np.column_stack(args)\n\tdata = np.split(data, np.array([8]), axis = 1)\n\tnumerical = data[0]\n\tcategorical = data[1]\n\tX = np.hstack((scaler.transform(numerical), enc.transform(categorical).toarray()))\n\treturn model.predict(X)";
}
}
