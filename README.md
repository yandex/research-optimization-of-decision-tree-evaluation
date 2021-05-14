Этот код является частью исследовательской работы на тему **"Оптимизации применения решающих деревьев с помощью SIMD инструкций"**

Для сборки и запуска в директори проекта выполнить
```
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release ../
cmake --build "build" --config Release
build/test_stand
```
