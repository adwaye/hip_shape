from OpenGL import GL as gl
from OpenGL import GLUT as glut
from OpenGL.arrays import vbo
import numpy as np

class VBOJiggle(object):

    def __init__(self,nvert=100,jiggliness=0.01):
        self.nvert = nvert
        self.jiggliness = jiggliness

        verts = 2*np.random.rand(nvert,2) - 1
        self.verts = np.require(verts,np.float32,'F')
        self.vbo = vbo.VBO(self.verts)

    def draw(self):

        gl.glClearColor(0,0,0,0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)

        self.vbo.bind()

        gl.glVertexPointer(2,gl.GL_FLOAT,0,self.vbo)
        gl.glColor(0,1,0,1)
        gl.glDrawArrays(gl.GL_LINE_LOOP,0,self.vbo.data.shape[0])

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        self.vbo.unbind()

        self.jiggle()

        glut.glutSwapBuffers()

    def jiggle(self):

        # jiggle half of the vertices around randomly
        delta = (np.random.rand(self.nvert//2,2) - 0.5)*self.jiggliness
        self.verts[:self.nvert:2] += delta

        # the data attribute of the vbo is the same as the numpy array
        # of vertices
        assert self.verts is self.vbo.data

        # # Approach 1:
        # # it seems like this ought to work, but it doesn't - all the
        # # vertices remain static even though the vbo's data gets updated
        # self.vbo.copy_data()

        # Approach 2:
        # this works, but it seems unnecessary to copy the whole array
        # up to the GPU, particularly if the array is large and I have
        # modified only a small subset of vertices
        self.vbo.set_array(self.verts)

if __name__ == '__main__':
    glut.glutInit()
    glut.glutInitDisplayMode( glut.GLUT_DOUBLE | glut.GLUT_RGB )
    glut.glutInitWindowSize( 250, 250 )
    glut.glutInitWindowPosition( 100, 100 )
    glut.glutCreateWindow( None )

    demo = VBOJiggle()
    glut.glutDisplayFunc( demo.draw )
    glut.glutIdleFunc( demo.draw )

    glut.glutMainLoop()