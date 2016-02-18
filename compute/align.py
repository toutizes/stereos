#!/opt/local/bin/python2.7
import sys
import numpy as np
import collections
import Image
import ImageDraw
import ImageFont

Point = collections.namedtuple('Point', ['x', 'y'])


def load_points(path):
  """Read a file containing point pairs.

  'path' contains 4 space-separated integers per line: x0 y0 x1 y1.

  Args:
    path: string.  path to open

  Returns:
    A list of pairs of points: [(Point(x, y), Point(x', y')), ...]
    The returned points use floats.
  """
  points = []
  with open(path) as f:
    for line in f.readlines():
      splits = line[:-1].split(' ')
      if len(splits) != 4:
        raise ValueError('Lines must contain 4 space-separated integers: %s' % line)
      points.append((Point(float(splits[0]), float(splits[1])), 
                     Point(float(splits[2]), float(splits[3]))))
  return points


def normalize_point(width, height, center, pt):
  """Map pt from image coords to normalized coords."""
  return Point((pt.x / width) - center.x, (pt.y / height) - center.y)


def denormalize_point(width, height, center, npt):
  """Map pt from normalized coords to image coords."""
  return Point((npt.x + center.x) * width, (npt.y + center.y) * height)


def normalize_points(width, height, points):
  c0 = Point(0.5, 0.5)
  c1 = Point(1.5, 0.5)
  return [(normalize_point(width, height, c0, p0),
           normalize_point(width, height, c1, p1))
          for p0, p1 in points]


def make_a(norm_points):
  data = [(p1.x * p0.x, p1.x * p0.y, p1.x,
           p1.y * p0.x, p1.y * p0.y, p1.y,
           p0.x, p0.y, 1.0)
          for p0, p1 in norm_points]
  return np.array(data).astype(np.float64)


def fundamental_matrix(A):
  U, s, V = np.linalg.svd(A, full_matrices=True)
  f = np.transpose(V)[:, -1]
  # print f
  # This should be close enough to 0.0
  print "Af =", np.linalg.norm(np.dot(A, f))
  F = np.transpose(f.reshape([3, 3]))
  # print F.shape
  # print F, np.linalg.matrix_rank(F)
  # Enforce F to be rank 2
  U, s, V = np.linalg.svd(F, full_matrices=True)
  S = np.diag(s)
  # print S
  S[2, 2] = 0.0
  F = np.dot(U, np.dot(S, V))
  # print F, np.linalg.matrix_rank(F)
  return F


def p0_for_p1(p1, F):
  pass

IMAGE = "/Users/matthieu/Pictures/Stereos/2016/2016-02-18/DSCF2314.jpg"


def draw_cross(draw, p, label, font):
  l = 100
  w = 4
  draw.line((p.x - l, p.y, p.x + l, p.y), width=w, fill="#ff0000")
  draw.line((p.x, p.y - l, p.x, p.y + l), width=w, fill="#ff0000")
  draw.text((p.x + 10, p.y + 10), label, fill="#ff0000", font=font)


def y_for_x(line, x):
  return (-line[2] - line[0] * x) / line[1]


def epipolar_line(F, np1):
  line = np.dot(F, np.array([np1.x, np1.y, 1]))
  return [Point(x, y_for_x(line, x)) for x in [-0.5, 0.5]]


def F_error(F, np0, np1):
  line = np.dot(F, np.array([np1.x, np1.y, 1]))
  return abs(np.dot(np.transpose([np0.x, np0.y, 1]), line))


def main():
  pts = load_points('/tmp/x.pts')
  font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 96)
  best_error = 1000.0
  best_F = None
  best_start = None
  for start in range(0, len(pts) - 8):
    im = Image.open(IMAGE)
    draw = ImageDraw.Draw(im)
    for i, (p0, p1) in enumerate(pts):
      draw_cross(draw, p0, str(i), font)
      draw_cross(draw, p1, str(i), font)
    width = im.size[0] / 2
    height = im.size[1]
    norm_pts = normalize_points(width, height, pts)
    a = make_a(norm_pts[start:start + 8])
    F = fundamental_matrix(a)
    c0 = Point(0.5, 0.5)
    if start > len(norm_pts) - 8:
      raise ValueError("not enough points")
    error = 0.0
    for np0, np1 in norm_pts:
      line = [denormalize_point(width, height, c0, np) 
              for np in epipolar_line(F, np1)]
      draw.line(line, width=3, fill="#00ff00")
      error += F_error(F, np0, np1)
    print "Error:", error
    if error < best_error:
      best_error = error
      best_F = F
      best_start = start
    im.save('/tmp/x-%s.jpg' % start)
  print "Best: %d (%g)" % (best_start, best_error)


main()
# a = np.array([[1, 0, 0, 0, 2],
#               [0, 0, 3, 0, 0],
#               [0, 0, 0, 0, 0],
#               [0, 4, 0, 0, 0]]).astype(np.float64)
# print a.shape
# print np.linalg.pinv(a)
  
