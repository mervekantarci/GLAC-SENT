"""
Not modified version can be found at:
https://github.com/tkim-snu/GLACNet
"""
import json

"""
Given json files of VIST dataset, we match annotations and image sequences
"""

class VIST:
    def __init__(self, sis_file = None, dii_file = None):
        sis_dataset = None
        dii_dataset = None
        if sis_file is not None:
            sis_dataset = json.load(open(sis_file, 'r'))
        else:
            print("sis file not provided!!!!!!")
        if  dii_file is not None:
            dii_dataset = json.load(open(dii_file, 'r'))
        self.LoadAnnotations(sis_dataset, dii_dataset)


    def LoadAnnotations(self, sis_dataset = None, dii_dataset = None):
        images = {}
        stories = {}

        # obtain image file names
        if 'images' in sis_dataset:
            for image in sis_dataset['images']:
                images[image['id']] = image

        # obtain annotations for each story sequence
        if 'annotations' in sis_dataset:
            annotations = sis_dataset['annotations']
            for annotation in annotations:
                story_id = annotation[0]['story_id']
                stories[story_id] = stories.get(story_id, []) + [annotation[0]]

        # some sequences have less than 5 images, in this work we don't need them
        ids = list(stories.keys())
        for story_id in ids:
            if len(stories[story_id]) < 5:
                del stories[story_id]

        self.images = images
        self.stories = stories

