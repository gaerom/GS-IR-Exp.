# Stage 1 - initial stage (add normal)
i=7
python train.py -m ./outputs/rwavs/$i -s ./release/$i --iterations 30000 --eval


# Stage 2 - Decomposition stage 
# python train.py -m ./outputs/rwavs/$i -s ./release/$i --start_checkpoint ./outputs/rwavs/$i/chkpnt30000.pth --iterations 30000 --eval --gamma --indirect

# Evaluation
# python render.py -m ./outputs/rwavs/$i -s ./release/$i --checkpoint ./outputs/rwavs/$i/chkpnt30000.pth --eval --skip_train --pbr