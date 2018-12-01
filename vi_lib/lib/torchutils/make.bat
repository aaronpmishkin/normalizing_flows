@ECHO OFF

rmdir build /S
python setup.py sdist bdist_wheel
twine upload dist/*
