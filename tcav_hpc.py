# SKAL LIGGE I ØVERSTE MAPPE
import json
import argparse

import tcav.activation_generator as act_gen

import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument("--num_random_exp", type=int, help="The number of random experiments, the same as the number of random folders")
    parser.add_argument("--save_filename", type=str, help="The name/path to save the results to. DONT ADD .JSON!")
    parser.add_argument("--working_dir", type=str, help="The path to the /tcav folder. i.e. /zhome/c9/156514/Desktop/tcav-master/tcav")
    parser.add_argument("--model_to_run", type=str, nargs="?", const="GoogleNet", help="The model to run, defaults to GoogleNet")
    parser.add_argument("--target", type=str, nargs="?", const="zebra", help="The taget of the model, defaults to zebra")
    args = parser.parse_args()

    model_to_run = args.model_to_run
    working_dir = args.working_dir
    
    # where activations are stored (only if your act_gen_wrapper does so)
    activation_dir =  working_dir + '/activations/'
    cav_dir = working_dir + '/cavs/'
    source_dir = working_dir + '/tcav_examples/image_models/imagenet/folder01'

    #TODO: add as a arg
    bottlenecks = ['mixed3a','mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']

    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(working_dir)
    utils.make_dir_if_not_exists(cav_dir)

    # this is a regularizer penalty parameter for linear classifier to get CAVs. 
    alphas = [0.1]   

    target = args.target
    concepts = ["dotted","striped","zigzagged"] #TODO: add as a arg

    sess = utils.create_session()


    # GRAPH_PATH is where the trained model is stored.
    
    GRAPH_PATH = working_dir + "/tcav_examples/image_models/imagenet/folder01/inception5h/tensorflow_inception_graph.pb"

    LABEL_PATH = working_dir + "/tcav_examples/image_models/imagenet/folder01/inception5h/imagenet_comp_graph_label_strings.txt"

    mymodel = model.GoogleNetWrapper_public(sess, GRAPH_PATH, LABEL_PATH)

    act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=100)


    num_random_exp = args.num_random_exp

    mytcav = tcav.TCAV(
        sess,
        target,
        concepts,
        bottlenecks,
        act_generator,
        alphas,
        cav_dir=cav_dir,
        num_random_exp=num_random_exp
    )
    
    print('This may take a while... Go get corny!')
    results = mytcav.run(run_parallel=True)
    print('done!')


    with open(f"{args.save_filename}.json", "w") as outfile:
        json.dump(results, outfile)
