# 0 percent

testmode='0'
# Change testmode to '1' for a light run to test your setup
ARCHITECTURES=('[128,512,256,128,512,256]')
RUNPARAMS=('[20000,512]')
LRS=('0.0002')
LATENTDIMS=('150')
DROPOUTS=('0.6')
MOMENTUMS=('0.6')
ALPHAS=('0.1')
PERCENTS=('0')
BETAS=('0.5')
# BETAS is beta1
BETASS=('0.99')
# BETASS is beta2


# i is the counter for repeating the experiment. You can set its range to a single number to only run the experiment once.
for i in 0 1 2 3 4 5 6 7 8 9; do
for percent in "${PERCENTS[@]}"; do
	for architecture in "${ARCHITECTURES[@]}"; do
		for runparam in "${RUNPARAMS[@]}"; do
			for lr in "${LRS[@]}"; do
				for latentdim in "${LATENTDIMS[@]}"; do
					for alpha in "${ALPHAS[@]}"; do
						for dropout in "${DROPOUTS[@]}"; do
							for momentum in "${MOMENTUMS[@]}"; do
								for beta in "${BETAS[@]}"; do
									for betaa in "${BETASS[@]}"; do
										python3.6 main.py ${percent} ${architecture} ${runparam} ${lr} ${latentdim} ${testmode} ${alpha} ${dropout} ${momentum} ${beta} ${betaa}
										sleep 1
									done
								done
							done
						done
					done
				done
			done
		done
	done
done
done
