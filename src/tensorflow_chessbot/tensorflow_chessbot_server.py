from flask import Flask, json, request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Ignore Tensorflow INFO print messages
import tensorflow as tf
import numpy as np
from .helper_functions import shortenFEN, unflipFEN
from . import helper_image_loading
from . import chessboard_finder
# from helper_functions import shortenFEN, unflipFEN
# import helper_image_loading
# import chessboard_finder
from .tensorflow_chessbot import ChessboardPredictor

api = Flask(__name__)


def main():
  api.run() 


@api.route('/fen', methods=['GET'])
def get_fen():
  filepath = request.args["filepath"]
  active = request.args["active"]

  img = helper_image_loading.loadImageFromPath(filepath)
  # Exit on failure to load image
  if img is None:
    raise Exception('Couldn\'t load file: "%s"' % filepath)
    
  # Resize image if too large
  # img = helper_image_loading.resizeAsNeeded(img)

  # Look for chessboard in image, get corners and split chessboard into tiles
  tiles, corners = chessboard_finder.findGrayscaleTilesInImage(img)

  # Exit on failure to find chessboard in image
  if tiles is None:
    # raise Exception('Couldn\'t find chessboard in image')
    return

  # Initialize predictor, takes a while, but only needed once
  predictor = ChessboardPredictor()
  fen, tile_certainties = predictor.getPrediction(tiles)
  predictor.close()
  
  short_fen = shortenFEN(fen)
  # Use the worst case certainty as our final uncertainty score
  certainty = tile_certainties.min()

  fen_rtn = "%s %s - - 0 1" % (short_fen, active)
  print(fen_rtn)
  return "%s %s - - 0 1" % (short_fen, active)

if __name__ == '__main__':
  api.run() 