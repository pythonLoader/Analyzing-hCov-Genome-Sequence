#!/bin/sh

cd "MAW based Phylogeny"

# -------- MAW Installation if not installed ------------ #

if [ ! -d "maw" ]; then

    while true; do
        read -p "You don't seem to have MAW here. Do you wish to install it? (Y/N) : " yn
        case $yn in
            [Yy]* ) break;;
            [Nn]* ) exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done

    echo "Installation starts.."
    git clone https://github.com/solonas13/maw.git
    cd maw
    ./pre-install.sh
    sudo make -f Makefile.64-bit.gcc     #this makefile doesn't compile without sudo
    echo "Installation done.."
    cd ..

fi


# --------------  Running MAW ----------------------------- #

if [ $1 = "dist" ]
then
    rm -f input.maw.fasta
    maw/maw -a DNA -i "$5" -o input.maw.fasta -k $2 -K $3       #prodcues MAW
    g++ main.cpp -std=c++0x -o main                             #compiles dist file  
    ./main $4                                                   #produces distance matrix
elif [ $1 = "tree" ]
then
    python distToTree.py $2                                     #produces tree
else
    echo "Error! Wrong command"
fi