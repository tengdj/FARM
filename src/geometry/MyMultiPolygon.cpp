#include "../include/MyPolygon.h"

MyMultiPolygon *MyMultiPolygon::read_multipolygon(){
	string input_line;
	getline(cin, input_line);
	if(input_line.size()==0){
		return NULL;
	}
	return new MyMultiPolygon(input_line.c_str());
}

MyPolygon *MyMultiPolygon::read_one_polygon(){

	MyMultiPolygon *mp = read_multipolygon();
	if(mp==NULL){
		return NULL;
	}
	MyPolygon *ret = mp->polygons[0]->clone();

	delete mp;
	return ret;
}

bool MyMultiPolygon::validate_wkt(string &wkt_str){
	const char *wkt = wkt_str.c_str();
	size_t len = wkt_str.size();
	size_t offset = 0;
	// validate the symbol MULTIPOLYGON
	while(offset<len &&wkt[offset]!='M'&&wkt[offset]!='P'){
		offset++;
	}
	if(offset==len){
		return false;
	}
	bool is_multiple = (wkt[offset]=='M');
	if(is_multiple){
		for(int i=0;i<strlen(multipolygon_char);i++){
			if(wkt[offset++]!=multipolygon_char[i]){
				return false;
			}
		}
		skip_space(wkt,offset);
		// read the left parenthesis
		if(wkt[offset++]!='('){
			return false;
		}
	}else {
		for(int i=0;i<strlen(polygon_char);i++){
			if(wkt[offset++]!=polygon_char[i]){
				return false;
			}
		}
		skip_space(wkt,offset);
	}
	// read the first polygon
	if(!MyPolygon::validate_polygon(wkt,offset,len)){
		return false;
	}
	if(is_multiple){
		skip_space(wkt, offset);
		// read the rest polygons
		while(offset<len && wkt[offset]==','){
			offset++;
			if(!MyPolygon::validate_polygon(wkt,offset,len)){
				return false;
			}
			skip_space(wkt,offset);
		}
		// read the right parenthesis
		if(wkt[offset++]!=')'){
			return false;
		}
	}
	return true;
}

MyMultiPolygon::MyMultiPolygon(const char *wkt){
	size_t offset = 0;
	// read the symbol MULTIPOLYGON
	while(wkt[offset]!='M'&&wkt[offset]!='P'){
		offset++;
	}
	bool is_multiple = (wkt[offset]=='M');
	if(is_multiple){
		for(int i=0;i<strlen(multipolygon_char);i++){
			assert(wkt[offset++]==multipolygon_char[i]);
		}
		skip_space(wkt,offset);
		// read the left parenthesis
		assert(wkt[offset++]=='(');
	}else {
		for(int i=0;i<strlen(polygon_char);i++){
			assert(wkt[offset++]==polygon_char[i]);
		}
		skip_space(wkt,offset);
	}
	// read the first polygon
	polygons.push_back(MyPolygon::read_polygon(wkt,offset));
	if(is_multiple){
		skip_space(wkt, offset);
		// read the rest polygons
		while(wkt[offset]==','){
			offset++;
			MyPolygon *in = MyPolygon::read_polygon(wkt,offset);
			assert(in);
			polygons.push_back(in);
			skip_space(wkt,offset);
		}
		// read the right parenthesis
		assert(wkt[offset++]==')');
	}
}

MyMultiPolygon::~MyMultiPolygon(){
	for(MyPolygon *p:polygons){
		delete p;
	}
}

void MyMultiPolygon::print(){
	cout<<"MULTIPOLYGON (";
	for(int i=0;i<polygons.size();i++){
		if(i>0){
			cout<<",";
		}
		polygons[i]->print_without_head();
	}
	cout<<")"<<endl;
}

void MyMultiPolygon::print_separate(){
	for(MyPolygon *p:polygons){
		p->print();
	}
}

void print_boxes(vector<box *> boxes){
	MyMultiPolygon *cboxes = new MyMultiPolygon();
	for(box *p:boxes){
		MyPolygon *m = MyPolygon::gen_box(*p);
		cboxes->insert_polygon(m);
	}
	cboxes->print();
	delete cboxes;
}



void MyMultiPoint::insert(vector<Point *> &origin_points){
	points.insert(points.begin(), origin_points.begin(), origin_points.end());
}
void MyMultiPoint::insert(Point *p){
	points.push_back(p);
}
void MyMultiPoint::print(size_t max_num){
	double sample_rate = 1.0;
	if(points.size()>max_num){
		sample_rate = 1.0*max_num/points.size();
	}
	printf("MULTIPOINT (");
	bool print_head = false;
	for(size_t i=0;i<points.size();i++){
		if(tryluck(sample_rate)){
			if(!print_head){
				print_head = true;
			}else{
				printf(",");
			}
			printf("%f %f",points[i]->x, points[i]->y);
		}
	}
	printf(")\n");
}


