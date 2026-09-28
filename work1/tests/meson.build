
test_exe = executable(

  'vector_test',

  'src/vector_test.cu',

  dependencies: core_dep,

  include_directories: [

    include_directories('../core/include'),

    include_directories('../core/src'),

  ],

)



test('vector_test', test_exe)

