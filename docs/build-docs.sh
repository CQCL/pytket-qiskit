rm -rf build/

mv pytket-docs-theming/_static .
mv pytket-docs-theming/quantinuum-sphinx .
mv pytket-docs-theming/conf.py .

sphinx-build -b html . build 

mv _static pytket-docs-theming
mv quantinuum-sphinx pytket-docs-theming 
mv conf.py pytket-docs-theming