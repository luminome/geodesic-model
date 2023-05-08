import * as THREE from 'three';
import * as util from 'three-sac/ui-util.js';
import loader from 'three-sac/ui-basic-loader.js';
import config from './config.json';
import Voronoi from 'voronoi';

const voronoi = new Voronoi();
const vc = {
    a: new THREE.Vector3(0, 0, 0),
    b: new THREE.Vector3(0, 0, 0),
    c: new THREE.Vector3(0, 0, 0),
    d: new THREE.Vector3(0, 0, 0),
    e: new THREE.Vector3(0, 0, 0)
};
const color_gradient = {
    c_arr:[[0,0,1],[0.5,0,0],[1,1,0],[0,1,1],[1,1,1]],
    get(pos){
        const L = color_gradient.c_arr.length-1;
        const i = Math.floor(pos*(L));
        const b = 1.0-((pos*(L))-i);
        const a = ((pos*(L))-i);
        const c = [0,0,0];
        for(let r = 0; r < 3; r++){
            c[r] = i === L ? color_gradient.c_arr[L][r] : (color_gradient.c_arr[i][r]*b + (color_gradient.c_arr[i+1][r]*a));
        }
        return c;
    }
}
const static_material = new THREE.MeshLambertMaterial({
    color: 0xFFFFFF
});
const faceFunctions = {
    get_centroid(vert_obj){
        const F = this.attrib;
        F.centroid = new THREE.Vector3();
        for(let n = 0; n<3; n++){
            const caret = [
                vert_obj[F.indices[n]*3],
                vert_obj[F.indices[n]*3+1],
                vert_obj[F.indices[n]*3+2],
            ];
            vc.e.fromArray(caret);
            F.centroid.add(vc.e);
        }
        F.centroid.divideScalar(3);
        return this;
    },
    get_grid(which='centroid'){
        const F = this.attrib;
        const polar = util.polar_mapper.cartesian2polar(F.centroid);
        const s = util.polar_mapper.polar2canvas(polar);
        F.coord = new THREE.Vector2(s.x, s.y);
        return this;
    },
    toString(){
        return `${this.attrib.name}-${this.attrib.index} ${this.attrib.indices.join()}`;
    }
}
function Face(indices, face_index=null){
    this.attrib = {
        index:face_index,
        indices:indices,
        name:'face-man',
    }
}

Object.assign(Face.prototype, faceFunctions);

const icosphere = (order = 0) => {
    //https://observablehq.com/@mourner/fast-icosphere-mesh
    // set up a 20-triangle icosahedron
    const f = (1 + 5 ** 0.5) / 2;
    const T = 4 ** order;

    const vertices = new Float32Array((10 * T + 2) * 3);

    vertices.set(Float32Array.of(
        -1, f, 0, 1, f, 0, -1, -f, 0, 1, -f, 0,
        0, -1, f, 0, 1, f, 0, -1, -f, 0, 1, -f,
        f, 0, -1, f, 0, 1, -f, 0, -1, -f, 0, 1));
    let triangles = Uint32Array.of(
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
        11, 10, 2, 5, 11, 4, 1, 5, 9, 7, 1, 8, 10, 7, 6,
        3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
        9, 8, 1, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7);

    let v = 12;
    const midCache = order ? new Map() : null; // midpoint vertices cache to avoid duplicating shared vertices

    function addMidPoint(a, b) {
        const key = Math.floor((a + b) * (a + b + 1) / 2) + Math.min(a, b); // Cantor's pairing function
        let i = midCache.get(key);
        if (i !== undefined) {
            midCache.delete(key);
            return i;
        }
        midCache.set(key, v);
        for (let k = 0; k < 3; k++) vertices[3 * v + k] = (vertices[3 * a + k] + vertices[3 * b + k]) / 2;
        i = v++;
        return i;
    }

    let trianglesPrev = triangles;
    for (let i = 0; i < order; i++) {
        // subdivide each triangle into 4 triangles
        triangles = new Uint32Array(trianglesPrev.length * 4);
        for (let k = 0; k < trianglesPrev.length; k += 3) {
            const v1 = trianglesPrev[k];
            const v2 = trianglesPrev[k + 1];
            const v3 = trianglesPrev[k + 2];
            const a = addMidPoint(v1, v2);
            const b = addMidPoint(v2, v3);
            const c = addMidPoint(v3, v1);
            let t = k * 4;
            triangles[t++] = v1;
            triangles[t++] = a;
            triangles[t++] = c;
            triangles[t++] = v2;
            triangles[t++] = b;
            triangles[t++] = a;
            triangles[t++] = v3;
            triangles[t++] = c;
            triangles[t++] = b;
            triangles[t++] = a;
            triangles[t++] = b;
            triangles[t++] = c;
        }
        trianglesPrev = triangles;
    }
    // normalize vertices
    for (let i = 0; i < vertices.length; i += 3) {
        const m = 1 / Math.hypot(vertices[i], vertices[i + 1], vertices[i + 2]);
        vertices[i] *= m;
        vertices[i + 1] *= m;
        vertices[i + 2] *= m;
    }
    return {vertices, triangles};
}


const model_loader = {
    count: 0,
    queue_length: null,
    bytes_loaded: 0,
    delta_time: 0,
    complete(resources, cat){
        let message = `(${model_loader.queue_length}) ${cat} loaded (${util.formatBytes(model_loader.bytes_loaded)}) in ${util.formatMs(model_loader.delta_time.stop())}.`;
        wedge.loader.messages(message);
        wedge.loader.model_post_load(resources);
    },
    status(count, obj){
        let message;
        if(count === 1){
            obj.t = util.timer(obj.id).start();
            message = `${obj.id+1} : ${obj.url} : ${obj.size}`;
        }else{
            model_loader.bytes_loaded += Number(obj.size);
            const t = obj.t.stop();
            message = `${(model_loader.queue_length-model_loader.count)+1}/${model_loader.queue_length} : ${obj.url} : ${util.formatBytes(obj.size)} : ${util.formatMs(t)}.`;
        }
        model_loader.count += count;
        wedge.loader.messages(message);
    },
    run(){
        model_loader.delta_time = util.timer('delta_time').start();
        const queue = wedge.config.sources.map((ca, id) => {
            return {url: `${ca[1]}`, variable: ca[0], size:'loading', type: 'json', cat: 'sources', id:id};
        });
        model_loader.queue_length = queue.length;
        loader(queue, model_loader.status).then(r => model_loader.complete(r, 'sources') );
    }
}


const wedge = {
    config: config,
    intersection_filter(intersection){
        wedge.trace.num = intersection.faceIndex;
        wedge.trace.pos = intersection.face.normal.toArray();
        wedge.trace.ind = [
            intersection.face.a,
            intersection.face.b,
            intersection.face.c
        ];
    },
    ready: false,
    trace:{},
    object: new THREE.Group(),
    model:{
        arrows:{},
        make_pointer_arrow(){
            let points = [];
            const raw_points = [
                0, 0,
                10, 20,
                6, 20,
                7, 32,
                0, 32
            ];

            for (let e = 0; e < raw_points.length; e += 2) {
                points.push(new THREE.Vector2(raw_points[e], raw_points[e + 1]));
            }

            const a_geometry = new THREE.LatheGeometry(points, 20);
            const a_material = new THREE.MeshStandardMaterial({
                color: 0xFFFFFF,
                side: THREE.FrontSide,
                flatShading: true,
                roughness: 0,
                metalness: 0.25,
                emissive: 0x161616
            });
            //const a_material = new MeshStandardMaterial( { color: 0xffff00 } );
            a_geometry.translate(0, -32, 0);
            a_geometry.rotateX(Math.PI / 2);
            a_geometry.scale(0.02, 0.02, 0.02);
            a_geometry.name = 'make_pointer_arrow';
            const arrow = new THREE.Mesh(a_geometry, a_material);
            arrow.userData.base_color = arrow.material.color;
            arrow.material.needsUpdate = true;

            return arrow;
        },
        make_dashed_line(radius) {
            const curve = new THREE.EllipseCurve(
                0, 0,            // ax, aY
                radius, radius,           // xRadius, yRadius
                0, 2 * Math.PI,  // aStartAngle, aEndAngle
                true,            // aClockwise
                0                 // aRotation
            );
            curve.updateArcLengths();

            const L = curve.getLength();

            const points = curve.getPoints(359);
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineDashedMaterial({
                color: 0x00FF00,
                linewidth: 1,
                scale: 1,
                dashSize: 1.5/L,
                gapSize: 1.5/L,
            });
            // Create the final object to add to the scene
            const line = new THREE.Line(geometry, material);
            line.userData.radius = radius;
            line.computeLineDistances();
            return line;
        },
        make_shells(){
            wedge.model.north_south = new THREE.Group();
            wedge.model.arrows.NS = wedge.model.make_pointer_arrow();
            wedge.model.arrows.NS.rotateX(Math.PI/-2);
            wedge.model.arrows.NS.position.set(0, wedge.abstract.scale+1.25, 0);
            wedge.model.north_south.add(wedge.model.arrows.NS);

            wedge.model.arrows.zNS = wedge.model.make_pointer_arrow();
            //wedge.model.arrows.zNS.rotateX(Math.PI/-2);
            wedge.model.arrows.zNS.position.set(0, 0, wedge.abstract.scale+1.25);
            wedge.model.north_south.add(wedge.model.arrows.zNS);


            const NS = wedge.model.make_dashed_line(wedge.abstract.scale+1);
            NS.rotateX(Math.PI / 2);
            wedge.model.north_south.add(NS);

            wedge.model.east_west = new THREE.Group();
            wedge.model.arrows.EW = wedge.model.make_pointer_arrow();
            //wedge.model.arrows.EW.rotateY(Math.PI/-2);
            wedge.model.arrows.EW.position.set(0, 0, wedge.abstract.scale+1.25);
            wedge.model.east_west.add(wedge.model.arrows.EW);

            const GMT = wedge.model.make_dashed_line(wedge.abstract.scale+1);
            GMT.rotateY(Math.PI / 2);
            wedge.model.north_south.add(GMT);

            const EW = wedge.model.make_dashed_line(wedge.abstract.scale+1);
            EW.rotateY(Math.PI / 2);
            wedge.model.east_west.add(EW);

            wedge.model.base = new THREE.Group();
            wedge.model.north_south.add(wedge.model.east_west);
            wedge.model.base.add(wedge.model.north_south);

            wedge.object.add(wedge.model.base);

            return wedge.object;
        }
    },
    source:{
        scaling: [4,2],
        axes: [1,0],
        data_shape: config.default_data_shape,//inherent before scaling
        shape: null,
        raw_data: [],
        parsed_data: [],
    },
    abstract:{
        faces:[],
        indices:[],
        vertices:[],
        mappings:[],
        color_amt: config.bias,
        face_depth: 1.05,
        face_scale: 1.0,
        scale: config.scale,
        order: config.order,
        model_on: true
    },
    build:{
        get_grid_data_rect(at_rect, poly_columns){
            const f = [
                Math.floor(at_rect[0]*wedge.source.shape[0]),
                Math.floor(at_rect[1]*wedge.source.shape[1]),
                Math.ceil(at_rect[2]*wedge.source.shape[0]),
                Math.ceil(at_rect[3]*wedge.source.shape[1])
            ]
            const s = wedge.source.shape;
            const data_vertices = [];
            const offset = [(1/s[0])*0.5,(1/s[1])*0.5];
            for(let i=f[0]; i<f[2]; i++){
                for(let j=f[1]; j<f[3]; j++){
                    const point = {
                        x:(i/s[0])+offset[0],
                        y:(j/s[1])+offset[1],
                        row: i,
                        col: j
                    }
                    if(util.point_in_poly(point, poly_columns[0],poly_columns[1])){
                        data_vertices.push(point);
                    }
                }
            }
            return {value:0.0, grid_data:data_vertices.map(g=>[g.row, g.col])};
        },
        get_grid_cell_structure(){
            const points_set = [];
            wedge.abstract.faces.map((f,i) => {
                points_set.push([i, f.attrib.coord]);
                if(f.attrib.coord.x === 1){
                    points_set.push([i+'E', {x:0.0, y:f.attrib.coord.y}]);
                }
            });

            const points = points_set.map(f => f[1]);
            const bbox = { xl: 0, xr:1, yt: 0, yb:1.0 };
            const diagram = voronoi.compute(points, bbox);

            wedge.abstract.faces.map((f,i) => {
                const ref = diagram.cells.filter(c => c.site.x === f.attrib.coord.x && c.site.y === f.attrib.coord.y);
                f.attrib.diagram_index = ref[0].site.voronoiId;
            });

            //let real_index = 0;
            wedge.abstract.mappings = [];
            let hold_cell = null;

            diagram.cells.forEach((cell,index) => {
                if (cell && cell.halfedges.length > 2) {
                    const uni_cell_data = {
                        index:index,
                        rect:null,
                        grid_points:null,
                        poly:null,
                        edge_ref: null
                    }

                    const segments = cell.halfedges.map(edge => edge.getEndpoint());
                    uni_cell_data.poly = segments.map(f => [f.x, f.y]);

                    const coord_cols = [[],[]];
                    segments.map(f => {
                        coord_cols[0].push(f.x);
                        coord_cols[1].push(f.y);
                    });

                    uni_cell_data.rect = [
                        Math.min(...coord_cols[0]),
                        Math.min(...coord_cols[1]),
                        Math.max(...coord_cols[0]),
                        Math.max(...coord_cols[1]),
                    ]

                    const gridded = wedge.build.get_grid_data_rect(uni_cell_data.rect, coord_cols);
                    uni_cell_data.grid_points = gridded.grid_data;

                    if(cell.site.x === 0) {
                        hold_cell = {cell:cell, index:index};
                    }

                    if(cell.site.x === 1 && hold_cell.cell.site.y === cell.site.y){
                        uni_cell_data.edge_ref = hold_cell.index;
                    }

                    wedge.abstract.mappings.push(uni_cell_data);
                }
            });
        },
        buildModel(){
            //#//build and collect all vertex data
            wedge.source.shape = [
                wedge.source.data_shape[0] * (Math.pow(2, wedge.source.scaling[0])),
                wedge.source.data_shape[1] * (Math.pow(2, wedge.source.scaling[1]))
            ];

            const ref = icosphere(wedge.abstract.order);
            const vertices_s = [...ref.vertices];
            const indices_s = [...ref.triangles];
            const k_faces = util.split_buffer(ref.triangles, 3); ///unpack to reg array from Uint16
            let apc = (vertices_s.length/3);

            const added_indices = [];
            const added_vertices = [];

            for(let i=0; i < k_faces.length; i++) {
                //for number of existing faces.
                const face = new Face(k_faces[i],i);
                face.get_centroid(vertices_s).get_grid();
                wedge.abstract.faces.push(face);
                vc.c.copy(face.attrib.centroid);

                const add_points = [];
                const add_points_indices = [];

                for(let t = 0; t<3; t++){
                    const index = face.attrib.indices[t];
                    ///get attribute position of index
                    util.v3_from_buffer(vertices_s, index, vc.a);
                    ///get new point location
                    vc.d.subVectors(vc.a, vc.c).multiplyScalar(wedge.abstract.face_scale).add(vc.c).multiplyScalar(wedge.abstract.face_depth);
                    add_points.push(...[vc.d.x, vc.d.y, vc.d.z]);
                    add_points_indices.push(apc);
                    apc++;
                }

                //need six more index groups per face
                face.attrib.indices_i = [...add_points_indices];
                const a = face.attrib.indices;
                const b = [...add_points_indices];
                const r_index = [
                    b[0],b[1],b[2],
                    b[0],a[0],b[1],
                    a[1],b[1],a[0],
                    b[1],a[1],b[2],
                    a[2],b[2],a[1],
                    b[2],a[2],b[0],
                    a[0],b[0],a[2]
                ]

                added_indices.push(...r_index);
                added_vertices.push(...add_points);

            }

            for(let u=0; u<added_vertices.length; u++) vertices_s.push(added_vertices[u]);
            for(let u=0; u<added_indices.length; u++) indices_s.push(added_indices[u]);

            wedge.abstract.vertices = vertices_s;
            wedge.abstract.indices = indices_s;

            //#//build and collect all vertex data
            wedge.build.get_grid_cell_structure();

        },
        generate_test_data(){
            const sample = {data:[]};
            for (let i=0; i < wedge.source.data_shape[1];i++) {
                for (let j = 0; j < wedge.source.data_shape[0]; j++) {
                    sample.data.push(i*j);
                }
            }
            return sample;
        },
        download_new_model(link_target) {
            //#// builder function for wedge model or just load.
            const build_timer = util.timer('build-model').start();
            wedge.build.buildModel();
            [
                [{'mappings': wedge.abstract.mappings}, 'mappings.json'],
                [{'faces': wedge.abstract.faces}, 'faces.json'],
                [{'vertices': wedge.abstract.vertices}, 'vertices.json'],
                [{'indices': wedge.abstract.indices}, 'indices.json'],
            ].map(dl => {
                const a = util.obj_to_download(dl[0], dl[1], 'auto');
                if (a) link_target.appendChild(a);
            });
            console.log(build_timer.var_name, util.formatMs(build_timer.stop()));
        }

    },
    parser:{
        format_raw_sample(sample){
            const d = [wedge.source.data_shape[wedge.source.axes[0]], wedge.source.data_shape[wedge.source.axes[1]]];
            //const values_all = [];
            const grid = new Array(d[1]);
            for (let i=0;i<d[1];i++){
                grid[i] = new Array(d[0]);
                for (let j=0;j<d[0];j++) {
                    //const v = sample[j*d[1]+i];
                    //values_all.push(v);
                    grid[i][j] = sample[j*d[1]+i];
                }
            }
            return grid; //{grid:grid, values:values_all};
        },
        interpolate_res(grid, axis=0){
            const grid_temp = [];

            if(axis === 0) {
                for (let i = 0; i < grid.length; i++) {
                    grid_temp.push(grid[i]);
                    const nulls = new Array(grid[0].length);
                    nulls.fill(null);
                    grid_temp.push(nulls);
                }

                grid_temp.push(grid_temp[0]);

                for (let i=0;i<grid_temp.length;i++){
                    for (let j=0;j < grid_temp[i].length; j++) {
                        const v = grid_temp[i][j];
                        if(v === null){
                            grid_temp[i][j] = Math.round(((grid_temp[i-1][j]+grid_temp[i+1][j])/2.0)*100000)/100000;
                        }
                    }
                }
                grid_temp.pop();
            }

            if(axis === 1) {
                for (let i = 0; i < grid.length; i++) {
                    const st = [];
                    for (let j=0;j < grid[i].length; j++) {
                        st.push(grid[i][j]);
                        st.push(null);
                    }
                    st.push(grid[i][0]);
                    grid_temp.push(st);
                }

                for (let i=0;i<grid_temp.length;i++){
                    for (let j=0;j < grid_temp[i].length; j++) {
                        const v = grid_temp[i][j];
                        if(v === null){
                            grid_temp[i][j] = Math.round(((grid_temp[i][j-1]+grid_temp[i][j+1])/2.0)*100000)/100000;
                        }
                    }
                    grid_temp[i].pop();
                }
            }

            return grid_temp;
        },
        parse(incoming = null, time_stamp=null) {
            wedge.source.shape = [
                wedge.source.data_shape[0] * (Math.pow(2, wedge.source.scaling[0])),
                wedge.source.data_shape[1] * (Math.pow(2, wedge.source.scaling[1]))
            ];
            let deg = null;
            if (time_stamp !== null){
                const d_arr = time_stamp.split('-');
                const d = {
                    month: d_arr[0]-1,
                    day: d_arr[1],
                    year: d_arr[2],
                    hour: d_arr[3],
                    minute: d_arr[4]
                }
                const dd = new Date(d.year, d.month, d.day, d.hour, d.minute, 0.0, 0.0);
                deg = ((dd.getHours()*15.0)+((dd.getMinutes()+dd.getTimezoneOffset())*0.25));
            }

            if (incoming !== null) {
                const max = Math.max(...incoming.data);
                const source = {
                    time:time_stamp,
                    grid: wedge.parser.format_raw_sample(incoming.data),
                    maximum: max
                }

                let d_grid = source.grid;

                wedge.source.axes.map(s => {
                    const lim = wedge.source.scaling[s];
                    for(let n=0; n < lim; n++){
                       d_grid = wedge.parser.interpolate_res(d_grid, s);
                    }
                });

                if(wedge.config.gauss.on){
                    const filter_obj = util.gauss.filter(d_grid, wedge.config.gauss.sigma);
                    const d = [d_grid.length, d_grid[0].length];
                    for (let i=0;i<d[0];i++){
                        for (let j=0;j<d[1];j++) {
                            d_grid[i][j] = filter_obj.data[i*d[1]+j];
                        }
                    }
                }

                for(let lon=0; lon<d_grid.length; lon++){
                    d_grid[lon].reverse();
                }

                d_grid.reverse();



                wedge.source.raw_data.push(source);
                wedge.source.parsed_data.push({time:time_stamp, grid:d_grid, maximum: max, deg:deg});
            }

        },
    },
    get_object(){
        wedge.model.vertices = new Float32Array(wedge.abstract.vertices);
        wedge.model.prev_vertices = new Float32Array(wedge.abstract.vertices);
        wedge.model.raw_vertices = new Float32Array(wedge.abstract.vertices);
        wedge.model.colors = new Float32Array(wedge.abstract.vertices.length);
        wedge.model.colors.fill(1.0);

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(wedge.model.vertices, 3, false));
        geometry.setAttribute('normal', new THREE.BufferAttribute(wedge.model.vertices, 3, false));
        geometry.setAttribute('color', new THREE.BufferAttribute(wedge.model.colors, 3, true));

        // geometry.attributes.normal.setUsage(THREE.DynamicDrawUsage);
        // geometry.attributes.normal.needsUpdate = true;

        geometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
        geometry.attributes.position.needsUpdate = true;

        geometry.attributes.color.setUsage(THREE.DynamicDrawUsage);
        geometry.attributes.color.needsUpdate = true;

        const indices = new Uint32Array(wedge.abstract.indices);
        const k_index = new THREE.Uint32BufferAttribute(indices, 1);
        geometry.setIndex(k_index);

        const blank_mat = static_material.clone();
        blank_mat.flatShading = true;
        blank_mat.vertexColors = true;
        blank_mat.emissive.setHex(0x333333);

        wedge.model.mesh = new THREE.Mesh( geometry, blank_mat );
        wedge.model.mesh.name = 'wedge_model';
        wedge.model.mesh.trace_function = wedge.intersection_filter;
        wedge.model.mesh.scale.setScalar(wedge.abstract.scale);
        wedge.model.make_shells();
        if(wedge.abstract.model_on) wedge.model.east_west.add(wedge.model.mesh);
        return wedge.object;
    },
    set_state(index = 0){
        wedge.abstract.data_index = index;
        if(!wedge.ready) return;
        const color_ceiling = wedge.source.raw_data[index].maximum;

        for(let f=0;f<wedge.abstract.faces.length; f++){
            const face = wedge.abstract.faces[f];
            const mapping = wedge.abstract.mappings[face.attrib.diagram_index];
            let value = util.average(mapping.grid_points.map(pt => wedge.source.parsed_data[index].grid[pt[0]][pt[1]]));
            if(mapping.edge_ref !== null){
                const edge_mapping = wedge.abstract.mappings[mapping.edge_ref];
                const edge_value = util.average(edge_mapping.grid_points.map(pt => wedge.source.parsed_data[index].grid[pt[0]][pt[1]]));
                value = (value+edge_value)/2.0;
            }
            face.attrib.data_value = (value / color_ceiling);

            for(let n=0;n < face.attrib.indices_i.length;n++){
                const c_array = color_gradient.get(face.attrib.data_value*wedge.abstract.color_amt);
                util.set_buffer_at_index(wedge.model.colors, face.attrib.indices_i[n], c_array);
                vc.a.fromArray(util.get_buffer_at_index(wedge.model.raw_vertices, face.attrib.indices_i[n]));
                vc.a.multiplyScalar(1+(face.attrib.data_value*0.125));
                util.set_buffer_at_index(wedge.model.vertices, face.attrib.indices_i[n], vc.a.toArray());
            }
        }

        wedge.model.mesh.geometry.attributes.color.needsUpdate = true;
        wedge.model.mesh.geometry.attributes.position.needsUpdate = true;
    },
    set_data(index = 0){
        wedge.abstract.data_index = index;
        if(!wedge.ready) return;
        const color_ceiling = wedge.source.raw_data[index].maximum;
        wedge.abstract.faces.map((f,i)=>{
            const mapping = wedge.abstract.mappings[f.attrib.diagram_index];
            let value = util.average(mapping.grid_points.map(pt => wedge.source.parsed_data[index].grid[pt[0]][pt[1]]));
            if(mapping.edge_ref !== null){
                const edge_mapping = wedge.abstract.mappings[mapping.edge_ref];
                const edge_value = util.average(edge_mapping.grid_points.map(pt => wedge.source.parsed_data[index].grid[pt[0]][pt[1]]));
                value = (value+edge_value)/2.0;
            }
            f.attrib.data_value = (value / color_ceiling);
        })
    },
    set_colors(){
        if(!wedge.ready) return;
        wedge.abstract.faces.map(f =>{
            for(let n=0;n < f.attrib.indices_i.length;n++){
                const c_array = color_gradient.get(f.attrib.data_value*wedge.abstract.color_amt);
                util.set_buffer_at_index(wedge.model.colors, f.attrib.indices_i[n], c_array);
                vc.a.fromArray(util.get_buffer_at_index(wedge.model.raw_vertices, f.attrib.indices_i[n]));
                vc.a.multiplyScalar(1+(f.attrib.data_value*0.125));
                util.set_buffer_at_index(wedge.model.vertices, f.attrib.indices_i[n], vc.a.toArray());
            }
        });
    },
    update(){
        //on idle really.
        for(let f=0;f<wedge.abstract.faces.length; f++) {
            const face = wedge.abstract.faces[f];
            for (let n = 0; n < face.attrib.indices_i.length; n++) {
                vc.a.fromArray(util.get_buffer_at_index(wedge.model.vertices, face.attrib.indices_i[n]));
                vc.b.lerp(vc.a, 0.1);
                util.set_buffer_at_index(wedge.model.vertices, face.attrib.indices_i[n], vc.b.toArray());
            }

        }
    },
    loader: {
        messages: (msg) => {
            console.log(msg);
        },
        model_post_load: (resources) => {
            const model_timer = util.timer('delta_time').start();
            resources.map(r => {
                wedge.abstract[r.variable] = r.raw[r.variable];
            })

            // init_vars.model.add(Model.get_object());
            // //Model.model.north_south.rotateX(util.deg_to_rad(-23.5));
            // Model.model.mesh.rotateY(Math.PI/2);
            //
            // init_vars.data_index = 0;
            wedge.ready = true;
            wedge.loader.messages(['model build in', util.formatMs(model_timer.stop())]);
        },
        model_loader
    }
}

export default wedge;