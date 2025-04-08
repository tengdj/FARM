#include "../include/MyPolygon.h"
#include "../include/Ideal.h"

MyPolygon::~MyPolygon(){
	if(boundary){
		delete boundary;
	}
	for(VertexSequence *p:holes){
		if(p){
			delete p;
		}
	}
	holes.clear();
	if(mbr){
		delete mbr;
	}
	if(mer){
		delete mer;
	}
	if(qtree){
		delete qtree;
	}
	if(rtree){
		delete rtree;
	}
	if(triangles){
		delete []triangles;
	}
	if(convex_hull){
		delete convex_hull;
	}
}

box *MyPolygon::getMBB(){
	if(mbr){
		return mbr;
	}
	mbr = boundary->getMBR();
	return mbr;
}

box *MyPolygon::getMER(query_context *ctx){

	if(mer){
		return mer;
	}

	Ideal *ras = new Ideal();
	ras->mbr = new box(mbr);
	ras->boundary = new VertexSequence(boundary->num_vertices, boundary->p);
 	ras->rasterization(ctx->vpr);
	vector<int> interiors = ras->get_pixels(IN);
	if(interiors.size()==0){
		return NULL;
	}
	int loops = ctx->mer_sample_round;
	int maxcount = 0;
	box *max_mer = NULL;
	while(loops-->0){
		int sample = get_rand_number(interiors.size())-1;
		box *curmer = ras->extractMER(interiors[sample]);
		if(max_mer){
			if(max_mer->area()<curmer->area()){
				delete max_mer;
				max_mer = curmer;
			}else{
				delete curmer;
			}
		}else{
			max_mer = curmer;
		}
	}
	interiors.clear();
	mer = max_mer;
	delete ras;
	return mer;
}

VertexSequence *MyPolygon::get_convex_hull(){
	if(convex_hull==NULL){
		convex_hull = boundary->convexHull();
	}
	return convex_hull;
}

QTNode *MyPolygon::partition_qtree(const int vpr){

	pthread_mutex_lock(&qtree_partition_lock);
	if(qtree){
		pthread_mutex_unlock(&qtree_partition_lock);
		return qtree;
	}

	int num_boxes = get_num_vertices()/vpr;
	int level = 1;
	while(pow(4,level)<num_boxes){
		level++;
	}
	int dimx = pow(2,level);
	int dimy = dimx;

	Ideal *ras = new Ideal();
	ras->mbr = new box(mbr);
	ras->boundary = new VertexSequence(boundary->num_vertices, boundary->p);
	ras->init_raster(dimx, dimy);
    ras->rasterization();

	int box_count = 4;
	int cur_level = 1;

	qtree = new QTNode(*(this->getMBB()));
	std::stack<QTNode *> ws;
	qtree->split();
	qtree->push(ws);

	vector<QTNode *> level_nodes;

	//breadth first traverse
	query_context qc;
	while(box_count<num_boxes||!ws.empty()){
		if(cur_level>level){
			dimx *= 2;
			dimy *= 2;
			level = cur_level;
			delete ras;
			ras = new Ideal();
			ras->mbr = new box(mbr);
			ras->boundary = new VertexSequence(boundary->num_vertices, boundary->p);
			ras->init_raster(dimx, dimy);
			ras->rasterization();
		}

		while(!ws.empty()){
			QTNode *cur = ws.top();
			ws.pop();
			bool contained = false;
			if(!ras->MyRaster::contain(&cur->mbr, contained)){
				level_nodes.push_back(cur);
			}else if(contained){
				cur->interior = true;
			}else{
				cur->exterior = true;
			}
		}


		for(QTNode *n:level_nodes){
			if(box_count<num_boxes){
				n->split();
				n->push(ws);
				box_count += 3;
			}else{
				break;
			}
		}
		cur_level++;
		level_nodes.clear();
	}
	//assert(box_count>=num_boxes);

	delete ras;
	pthread_mutex_unlock(&qtree_partition_lock);

	return qtree;
}

MyPolygon *MyPolygon::clone(){
	MyPolygon *polygon = new MyPolygon();
	polygon->boundary = boundary->clone();
	for(VertexSequence *t:holes){
		polygon->holes.push_back(t->clone());
	}
	polygon->getMBB();
	return polygon;
}

size_t MyPolygon::get_data_size(){
	size_t ds = 0;
	ds += sizeof(size_t);
	// for boundary
	ds += boundary->get_data_size();
	for(VertexSequence *vs:holes){
		ds += vs->get_data_size();
	}
	return ds;
}

size_t MyPolygon::get_rtree_size(){
	if(rtree){
		int count = rtree->node_count();
		return count*4*8;
	}else{
		return 0;
	}
}

void MyPolygon::triangulate(){
	assert(!boundary->clockwise());
	if(triangle_num > 0){
		return;
	}
	assert(boundary->p);
	assert(boundary->num_vertices>0);

	vector<Vertex *> polyline = boundary->pack_to_polyline();
	assert(polyline.size() > 0);
	CDT *cdt = new CDT(polyline);
	cdt->Triangulate();
	vector<Triangle *> tri = cdt->GetTriangles();
	triangles = new Point[3*tri.size()];
	for(int i=0;i<tri.size();i++){
		triangles[i*3].x = tri[i]->point(0)->x;
		triangles[i*3].y = tri[i]->point(0)->y;
		triangles[i*3+1].x = tri[i]->point(1)->x;
		triangles[i*3+1].y = tri[i]->point(1)->y;
		triangles[i*3+2].x = tri[i]->point(2)->x;
		triangles[i*3+2].y = tri[i]->point(2)->y;
	}
	triangle_num = tri.size();
	delete cdt;
	for(Vertex *v:polyline){
		delete v;
	}
	polyline.clear();
}

VertexSequence *MyPolygon::read_vertices(const char *wkt, size_t &offset, bool clockwise){
	// read until the left parenthesis
	skip_space(wkt, offset);
	assert(wkt[offset++]=='(');

	// count the number of vertices
	int cur_offset = offset;
	int num_vertices = 0;
	while(wkt[cur_offset++]!=')'){
		if(wkt[cur_offset]==','){
			num_vertices++;
		}
	}
	num_vertices++;
	VertexSequence *vs = new VertexSequence(num_vertices);

	// read x/y
	for(int i=0;i<num_vertices;i++){
		vs->p[i].x = read_double(wkt, offset);
		vs->p[i].y = read_double(wkt, offset);
	}
	if(clockwise){
		if(!vs->clockwise()){
			vs->reverse();
		}
	}else{
		if(vs->clockwise()){
			vs->reverse();
		}
	}

	// read until the right parenthesis
	skip_space(wkt, offset);
	assert(wkt[offset++]==')');
	return vs;
}

MyPolygon *MyPolygon::read_polygon(const char *wkt, size_t &offset){

	MyPolygon *polygon = new MyPolygon();
	skip_space(wkt, offset);
	// left parentheses for the entire polygon
	assert(wkt[offset++]=='(');

	// read the vertices of the boundary polygon
	// the vertex must rotation in clockwise
	polygon->boundary = read_vertices(wkt, offset,false);
	if(polygon->boundary->clockwise()){
		polygon->boundary->reverse();
	}
	polygon->boundary->fix();
	skip_space(wkt, offset);
	//polygons as the holes of the boundary polygon
	while(wkt[offset]==','){
		offset++;
		VertexSequence *vc = read_vertices(wkt, offset,true);
		if(!vc->clockwise()){
			vc->reverse();
		}
		vc->fix();
		polygon->holes.push_back(vc);

		skip_space(wkt, offset);
	}
	assert(wkt[offset++]==')');
	polygon->getMBB();
	return polygon;
}

bool MyPolygon::validate_vertices(const char *wkt, size_t &offset, size_t &len){
	// read until the left parenthesis
	skip_space(wkt, offset);
	if(wkt[offset++]!='('){
		return false;
	}

	// count the number of vertices
	int cur_offset = offset;
	int num_vertices = 0;
	while(wkt[cur_offset++]!=')'){
		if(wkt[cur_offset]==','){
			num_vertices++;
		}
	}
	num_vertices++;

	// read x/y
	for(int i=0;i<num_vertices;i++){
		read_double(wkt, offset);
		read_double(wkt, offset);
	}

	// read until the right parenthesis
	skip_space(wkt, offset);
	if(wkt[offset++]!=')'){
		return false;
	}
	return true;
}

bool MyPolygon::validate_polygon(const char *wkt, size_t &offset, size_t &len){
	skip_space(wkt, offset);
	// left parentheses for the entire polygon
	if(wkt[offset++]!='('){
		return false;
	}

	// read the vertices of the boundary polygon
	// the vertex must rotation in clockwise
	if(!validate_vertices(wkt, offset, len)){
		return false;
	}
	skip_space(wkt, offset);
	//polygons as the holes of the boundary polygon
	while(wkt[offset]==','){
		offset++;
		if(!validate_vertices(wkt, offset, len)){
			return false;
		}
		skip_space(wkt, offset);
	}
	if(wkt[offset++]!=')'){
		return false;
	}
	return true;
}

MyPolygon *MyPolygon::gen_box(double min_x,double min_y,double max_x,double max_y){
	MyPolygon *mbr = new MyPolygon();
	mbr->boundary = new VertexSequence(5);
	mbr->boundary->p[0].x = min_x;
	mbr->boundary->p[0].y = min_y;
	mbr->boundary->p[1].x = max_x;
	mbr->boundary->p[1].y = min_y;
	mbr->boundary->p[2].x = max_x;
	mbr->boundary->p[2].y = max_y;
	mbr->boundary->p[3].x = min_x;
	mbr->boundary->p[3].y = max_y;
	mbr->boundary->p[4].x = min_x;
	mbr->boundary->p[4].y = min_y;
	return mbr;
}

MyPolygon *MyPolygon::gen_box(box &pix){
	return gen_box(pix.low[0],pix.low[1],pix.high[0],pix.high[1]);
}

bool MyPolygon::contain(Point &p){
    return boundary->contain(p);
}

bool MyPolygon::contain_rtree(RTNode *node, Point &p, query_context *ctx){
	if(!node->contain(p)){
		return false;
	}
	if(node->is_leaf()){
		assert(node->node_element);
		Point *triangle = (Point *)node->node_element;
		struct timeval start = get_cur_time();
		bool ret = false;
		for(int i=0;i<=2;i++){
			Point *start = triangle+i;
			Point *end = triangle+(i+1)%3;
			if((start->y >= p.y ) != (end->y >= p.y)){
				double xint = (end->x - start->x) * (p.y - start->y)/ (end->y - start->y) + start->x;
				if(p.x <= xint){
					ret = !ret;
				}
			}
		}
		return !ret;
	}
	for(RTNode *ch:node->children){
		if(contain_rtree(ch,p,ctx)){
			return true;
		}
	}
	return false;
}

double MyPolygon::distance_rtree(Point &p, query_context *ctx){
	assert(rtree);
	queue<RTNode *> pixq;
	for(RTNode *p:rtree->children){
		pixq.push(p);
	}
	// set this value as the MINMAXDIST
	ctx->distance = DBL_MAX;
	vector<std::pair<RTNode *, double>> tmp;
	while(pixq.size()>0){
		int s = pixq.size();
		for(int i=0;i<s;i++){
			RTNode *pix = pixq.front();
			pixq.pop();
			double mindist = pix->distance(p,ctx->geography);
			if(mindist>ctx->distance){
				continue;
			}
			double maxdist = pix->max_distance(p,ctx->geography);

			if(maxdist<ctx->distance){
				ctx->distance = maxdist;
			}

			tmp.push_back(std::pair<RTNode *, double>(pix, mindist));
		}

		for(std::pair<RTNode *, double> pr:tmp){
			if(pr.second<=ctx->distance){
				if(pr.first->children.size()==0){
					Point *triangle = (Point *)pr.first->node_element;
					for(int i=0;i<=2;i++){
						double dist = point_to_segment_distance(p, triangle[i], triangle[(i+1)%3],ctx->geography);
						if(ctx->distance>dist){
							ctx->distance = dist;
						}
					}
					ctx->edge_checked.counter += 3;
				}else{
					for(RTNode *p:pr.first->children){
						pixq.push(p);
					}
				}
			}
		}

		tmp.clear();
	}

	return ctx->distance;
}

// calculate the distance with rtree from a segment
double MyPolygon::distance_rtree(Point &start, Point &end, query_context *ctx){
	assert(rtree);

	// start a breadth first search
	queue<RTNode *> pixq;
	for(RTNode *p:rtree->children){
		pixq.push(p);
	}
	// set this value as the MINMAXDIST
	double mindist = DBL_MAX;
	vector<std::pair<RTNode *, double>> candidates;
	while(pixq.size()>0){
		for(int i=0;i<pixq.size();i++){
			RTNode *pix = pixq.front();
			pixq.pop();
			// this node and its descendants must not be candidate
			double dist = pix->distance(start, end, ctx->geography);
			if(dist >= mindist){
				continue;
			}
			// maximum possible distance between the target and this node
			double maxdist = pix->max_distance(start, end, ctx->geography);
			mindist = min(maxdist, mindist);
			candidates.push_back(std::pair<RTNode *, double>(pix, dist));
		}

		for(std::pair<RTNode *, double> pr:candidates){
			// minimum possible distance of this node
			// must be smaller than the current minimum
			if(pr.second<=mindist){
				// is leaf
				if(pr.first->is_leaf()){
					Point *triangle = (Point *)pr.first->node_element;
					for(int i=0;i<=2;i++){
						double dist = segment_to_segment_distance(start, end, triangle[i], triangle[(i+1)%3],ctx->geography);
						mindist = min(mindist, dist);
						// close enough for a within query
						if(ctx->within(mindist)){
							return mindist;
						}
					}
					ctx->edge_checked.counter += 3;
				}else{
					for(RTNode *p:pr.first->children){
						pixq.push(p);
					}
				}
			}
		}
		candidates.clear();
	}

	return mindist;
}

bool MyPolygon::contain(Point &p, query_context *ctx, bool profile){
	// the MBB may not be checked for within query
	if(!mbr->contain(p)){
		return false;
	}
	// check the maximum enclosed rectangle (MER)
	if(mer&&mer->contain(p)){
		return true;
	}
	// check the convex hull
	if(convex_hull&&!convex_hull->contain(p)){
		return false;
	}
	
	if(qtree){
		QTNode *tnode = get_qtree()->retrieve(p);
		assert(tnode->isleaf()&&tnode->mbr.contain(p));
		if(tnode->exterior) return false;
		else if(tnode->interior) return true;
	}

	bool contained = false;
	// refinement step
	if(ctx->perform_refine){
		if(rtree) contained = contain_rtree(rtree,p,ctx);
		else contained = contain(p);
	}

	return contained;
}

bool MyPolygon::contain(MyPolygon *target, query_context *ctx){
	if(!getMBB()->contain(*target->getMBB())){
		//log("mbb do not contain");
		return false;
	}

	if(qtree) {
		// filtering with the mbr of the target against the qtree
		bool isin = false;
		if(get_qtree()->determine_contain(*(target->getMBB()), isin)){
			return isin;
		}
		ctx->border_checked.counter++;
		// otherwise, checking all the edges to make sure no intersection
		if(segment_intersect_batch(boundary->p, target->boundary->p, boundary->num_vertices, target->boundary->num_vertices)){
			return false;
		}
	}

	// filtering with mer, mbr, and convex hull
	if(mer){
		// filter with the mer of source and mbr of the target
		if(mer->contain(*target->getMBB())){
			return true;
		}

		// filter with the convex hull of target
		if(target->convex_hull){
			Point mer_vertices[5];
			mer->to_array(mer_vertices);
			if(!segment_intersect_batch(mer_vertices, target->convex_hull->p, 5, target->convex_hull->num_vertices)){
				if(mer->contain(convex_hull->p[0])){
					return true;
				}
			}
		}
		// further filtering with the mbr of the target
		Point mbb_vertices[5];
		target->mbr->to_array(mbb_vertices);
		// no intersection between this polygon and the mbr of the target polygon
		if(!segment_intersect_batch(boundary->p, mbb_vertices, boundary->num_vertices, 5)){
			// the target must be the one which is contained (not contain) as its mbr is contained
			if(contain(mbb_vertices[0], ctx)){
				return true;
			}
		}
	}

	if(ctx->perform_refine){
		// use the internal rtree if it is created	
		if(rtree){
			for(int i=0;i<target->get_num_vertices();i++){
				if(!contain_rtree(rtree, *target->get_point(i), ctx)){
					return false;
				}
			}
			return true;
		}else{
			// otherwise, checking all the edges to make sure no intersection
			if(segment_intersect_batch(boundary->p, target->boundary->p, boundary->num_vertices, target->boundary->num_vertices)){
				return false;
			}
		}
	}

	// this is the last step for all the cases, when no intersection segment is identified
	// pick one point from the target and it must be contained by this polygon
	Point p(target->getx(0),target->gety(0));
	return contain(p, ctx,false);
}

double MyPolygon::distance(Point &p, query_context *ctx, bool profile){
	// distance is 0 if contained by the polygon
	double mindist = getMBB()->max_distance(p, ctx->geography);
	bool contained = contain(p, ctx, profile);
	if(contained) return 0;

	if(qtree){
		if(ctx->is_within_query()&&!qtree->within(p, ctx->within_distance)){
			return DBL_MAX;
		}
	}

	//checking convex
	if(ctx->is_within_query()&&convex_hull){
		double min_dist = DBL_MAX;
		for(int i=0;i<convex_hull->num_vertices-1;i++){
			double dist = point_to_segment_distance(p, convex_hull->p[i], convex_hull->p[i+1], ctx->geography);
			if(dist<min_dist){
				min_dist = dist;
			}
		}
		if(min_dist>ctx->within_distance){
			return min_dist;
		}
	}
	//SIMPVEC return
	if(ctx->perform_refine){
		if(rtree){
			return distance_rtree(p,ctx);
		}else{
			return boundary->distance(p, ctx->geography);
		}
	}else{
		return DBL_MAX;
	}
}

double MyPolygon::distance(MyPolygon *target, query_context *ctx, bool profile){
	if(qtree){
		// checking the qtree for filtering
		if(ctx->is_within_query()){
			for(int i=0;i<target->boundary->num_vertices-1;i++){
				// if any edge is within the distance after checking the QTree, get the exact distance from it to the source polygon
				if(qtree->within(target->boundary->p[i], target->boundary->p[i+1], ctx->within_distance)){
					double dist = segment_sequence_distance(boundary->p, target->boundary->p+i,
							boundary->num_vertices, 1,ctx->geography);
					if(dist <= ctx->within_distance){
						return dist;
					}
				}
			}
		}
	}


	//checking convex for filtering
	if(ctx->is_within_query() && convex_hull && target->convex_hull){
		double dist = segment_sequence_distance(convex_hull->p, target->convex_hull->p,
				convex_hull->num_vertices, target->convex_hull->num_vertices, ctx->geography);
		// the convex_hull is a conservative approximation of the original polygon,
		// so the distance between the convex hulls of two polygon is smaller than
		// the real distance
		if(dist>ctx->within_distance){
			return dist;
		}
	}

	//SIMPVEC return
	if(ctx->perform_refine){
		if(rtree){
			double mindist = DBL_MAX;
			for(int i=0;i<target->boundary->num_vertices-1;i++){
				double dist = distance_rtree(target->boundary->p[i], target->boundary->p[i+1], ctx);
				mindist = min(mindist, dist);
				//log("%f",mindist);

				if(ctx->within(mindist)){
					return mindist;
				}
			}
			return mindist;
		}else{
			return segment_sequence_distance(boundary->p, target->boundary->p,
									boundary->num_vertices, target->boundary->num_vertices,ctx->geography);
		}
	}
	assert(false);
	return DBL_MAX;
}

void MyPolygon::print_without_head(bool print_hole, bool complete_ring){
	assert(boundary);
	cout<<"(";

	boundary->print(complete_ring);
	if(print_hole){
		for(VertexSequence *vs:this->holes){
			cout<<", ";
			vs->print(complete_ring);
		}
	}
	cout<<")";
}

void MyPolygon::print(bool print_id, bool print_hole){
	// if(print_id){
	// 	cout<<"id:\t"<<this->id<<endl;
	// }
	cout<<"LINESTRING ";
	print_without_head(print_hole);
	cout<<endl;
}

PolygonMeta MyPolygon::get_meta(){
	PolygonMeta pmeta;
	pmeta.size = get_data_size();
	pmeta.num_vertices = get_num_vertices();
	pmeta.mbr = *getMBB();
	return pmeta;
}

size_t MyPolygon::encode(char *target){
	size_t encoded = 0;
	((size_t *)target)[0] = holes.size();
	encoded += sizeof(size_t); //saved one size_t for number of holes
	encoded += boundary->encode(target+encoded);
	for(VertexSequence *vs:holes){
		encoded += vs->encode(target+encoded);
	}
	return encoded;
}

size_t MyPolygon::decode(char *source){
	size_t decoded = 0;
	assert(!boundary);
	boundary = new VertexSequence();
	size_t num_holes = ((size_t *)source)[0];
	decoded += sizeof(size_t);
	decoded += boundary->decode(source+decoded);
	for(size_t i=0;i<num_holes;i++){
		VertexSequence *vs = new VertexSequence();
		decoded += vs->decode(source+decoded);
	}
	return decoded;
}

