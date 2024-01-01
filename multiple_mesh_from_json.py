import pygltflib
import json
import numpy as np

def convert_to_y_up(points):
    """Z-UpからY-Upへ座標変換"""
    return np.array([[x, z, y] for x, y, z in points])

with open("C:\\Users\\81803\\Desktop\\export_data3.json", 'r') as file:
    meshes_data = json.load(file)

pointsArray= []
normalsArray = []
facesArray = []
userDataArray = []

for mesh_data in meshes_data:
    # 頂点データを準備
    vertices = convert_to_y_up(mesh_data["vertices"])
    npArray = np.array(vertices, dtype="float32")
    pointsArray.append(npArray)

    # 法線データを準備
    normals = convert_to_y_up(mesh_data["normals"])  # 法線も座標変換が必要
    npNormals = np.array(normals, dtype="float32")
    normalsArray.append(npNormals)

    # 頂点のインデックスを準備
    faces = mesh_data["faces"]
    npFaces = np.array(faces, dtype="uint32")
    facesArray.append(npFaces)

    # userDataを準備
    userData = mesh_data["userData"]
    userDataArray.append(userData)


binary_blob = bytearray()
for points, normals,faces in zip(pointsArray, normalsArray, facesArray):

    points_binary_blob = points.tobytes()
    normals_binary_blob = normals.tobytes()
    faces_binary_blob = faces.flatten().tobytes()

    binary_blob.extend(faces_binary_blob)
    binary_blob.extend(points_binary_blob)
    binary_blob.extend(normals_binary_blob)

buffer = pygltflib.Buffer(byteLength=len(binary_blob))

# バッファビューの作成
bufferViews = []
byte_offset = 0
accessors = []

for i, (points, normals, faces) in enumerate(zip(pointsArray, normalsArray, facesArray)):
    faces_byte_length = len(faces.flatten().tobytes())
    bufferViews.append(pygltflib.BufferView(
        buffer=0,
        byteOffset=byte_offset,
        byteLength=faces_byte_length,
        target=pygltflib.ELEMENT_ARRAY_BUFFER
    ))
    byte_offset += faces_byte_length

    # 頂点位置データのバッファビュー
    points_byte_length = len(points.tobytes())
    bufferViews.append(pygltflib.BufferView(
        buffer=0,
        byteOffset=byte_offset,
        byteLength=points_byte_length,
        target=pygltflib.ARRAY_BUFFER
    ))
    byte_offset += points_byte_length

    # 法線データのバッファビュー
    normals_byte_length = len(normals.tobytes())
    bufferViews.append(pygltflib.BufferView(
        buffer=0,
        byteOffset=byte_offset,
        byteLength=normals_byte_length,
        target=pygltflib.ARRAY_BUFFER
    ))
    byte_offset += normals_byte_length

    accessors.extend([
        pygltflib.Accessor(
            bufferView=i * 3,
            componentType=pygltflib.UNSIGNED_INT,
            count=len(faces.flatten()),
            type=pygltflib.SCALAR,
            max=[int(faces.max())],
            min=[int(faces.min())],
        ),
        pygltflib.Accessor(
            bufferView=i * 3 + 1,
            componentType=pygltflib.FLOAT,
            count=len(points),
            type=pygltflib.VEC3,
            max=points.max(axis=0).tolist(),
            min=points.min(axis=0).tolist(),
        ),
        pygltflib.Accessor(
            bufferView=i * 3 + 2,
            componentType=pygltflib.FLOAT,
            count=len(normals),
            type=pygltflib.VEC3,
            max=normals.max(axis=0).tolist(),
            min=normals.min(axis=0).tolist(),
        )
    ])


# メッシュとノードの作成
meshes = []
nodes = []

for i, userData in enumerate(userDataArray):
    mesh_index = len(meshes)
    meshes.append(pygltflib.Mesh(
        primitives=[pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=i * 3 + 1,NORMAL=i * 3 + 2), indices=i * 3)]
    ))
    meshes[mesh_index].extras = {"id": userData}
    nodes.append(pygltflib.Node(mesh=mesh_index))

gltf = pygltflib.GLTF2(
    scene=0,
    scenes=[pygltflib.Scene(nodes=list(range(len(nodes))))],
    nodes=nodes,
    meshes=meshes,
    accessors=accessors,
    bufferViews=bufferViews,
    buffers=[buffer]
)

# バイナリブロブをGLTFに設定
gltf.set_binary_blob(bytes(binary_blob))


# GLTFファイルの保存
gltf.save("test.glb")