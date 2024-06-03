ob_define(PYTHON_DIR "/usr/local/python39")

set(PYTHON_LIB_DIR "${PYTHON_DIR}/lib")

message(STATUS "Set Python dir ${PYTHON_DIR}")

add_library(python_lib INTERFACE)
target_include_directories(
  python_lib INTERFACE
  "${PYTHON_DIR}/include/python3.9"
  "${PYTHON_LIB_DIR}"
  "${PYTHON_LIB_DIR}/python3.9/site-packages/numpy/core/include/")
target_link_libraries(python_lib INTERFACE
    -L/usr/local/python39/lib -lpython3.9 -lpthread -ldl  -lutil -lm  -Xlinker -export-dynamic)