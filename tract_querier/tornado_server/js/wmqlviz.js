$(function() {
	
	// create the sliders for the 2D sliders
	$("#yellow_slider").slider({
		slide: volumeslicingX
	});
	$("#yellow_slider .ui-slider-handle").unbind('keydown');
	
	$("#red_slider").slider({
		slide: volumeslicingY
	});
	$("#red_slider .ui-slider-handle").unbind('keydown');
	
	$("#green_slider").slider({
		slide: volumeslicingZ
	});
	$("#green_slider .ui-slider-handle").unbind('keydown');
	
});

function init_viewer2d() {

	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];

	// X Slice
	if (sliceX) {
		sliceX.destroy();
	}
	sliceX = new X.renderer2D();
	sliceX.container = 'sliceX';
	sliceX.orientation = 'X';
	sliceX.init();
	sliceX.add(volume);
	sliceX.render();
	sliceX.interactor.onMouseMove = function() {
		if (_ATLAS_.hover){
			clearTimeout(_ATLAS_.hover);
		}
		_ATLAS_.hover = setTimeout(on2DHover.bind(this, sliceX), 100);
	};

	// Y Slice
	if (sliceY) {
		sliceY.destroy();
	}
	sliceY = new X.renderer2D();
	sliceY.container = 'sliceY';
	sliceY.orientation = 'Y';
	sliceY.init();
	sliceY.add(volume);
	sliceY.render();
	sliceY.interactor.onMouseMove = function() {
		if (_ATLAS_.hover){
			clearTimeout(_ATLAS_.hover);
		}
		_ATLAS_.hover = setTimeout(on2DHover.bind(this, sliceY), 100);
	};

	// Z Slice
	if (sliceZ) {
		sliceZ.destroy();
	}
	sliceZ = new X.renderer2D();
	sliceZ.container = 'sliceZ';
	sliceZ.orientation = 'Z';
	sliceZ.init();
	sliceZ.add(volume);
	sliceZ.render();
	sliceZ.interactor.onMouseMove = function() {
		if (_ATLAS_.hover){
			clearTimeout(_ATLAS_.hover);
		}
		_ATLAS_.hover = setTimeout(on2DHover.bind(this, sliceZ), 100);
	  };

	// update 2d slice sliders
	var dim = volume.dimensions;
	$("#yellow_slider").slider("option", "disabled", false);
	$("#yellow_slider").slider("option", "min", 0);
	$("#yellow_slider").slider("option", "max", dim[0] - 1);
	$("#yellow_slider").slider("option", "value", volume.indexX);
	$("#red_slider").slider("option", "disabled", false);
	$("#red_slider").slider("option", "min", 0);
	$("#red_slider").slider("option", "max", dim[1] - 1);
	$("#red_slider").slider("option", "value", volume.indexY);
	$("#green_slider").slider("option", "disabled", false);
	$("#green_slider").slider("option", "min", 0);
	$("#green_slider").slider("option", "max", dim[2] - 1);
	$("#green_slider").slider("option", "value", volume.indexZ);
}//init_viewer2d()

// show labels on hover
function on2DHover(renderer) {

	// check that feature is enabled
	if (!_ATLAS_.hoverLabelSelect){
		return;
	}

	// get cursor position
	var mousepos = renderer.interactor.mousePosition;
	var ijk = renderer.xy2ijk(mousepos[0], mousepos[1]);
	if (!ijk) {
		return;
	}

	//
	var orientedIJK = ijk.slice();
	orientedIJK[0] = ijk[2];
	orientedIJK[1] = ijk[1];
	orientedIJK[2] = ijk[0];

	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	
	// get the number associated with the label
	var labelvalue = volume.labelmap.image[orientedIJK[0]][orientedIJK[1]][orientedIJK[2]];
	if (!labelvalue || labelvalue == 0) {
		return;
	}
	
	var labelname = volume.labelmap.colortable.get(labelvalue)[0];
	var _r = parseInt(volume.labelmap.colortable.get(labelvalue)[1] * 255, 10);
	var _g = parseInt(volume.labelmap.colortable.get(labelvalue)[2] * 255, 10);
	var _b = parseInt(volume.labelmap.colortable.get(labelvalue)[3] * 255, 10);
	$('#anatomy_caption').html(labelname);
	$('#anatomy_caption').css('color', 'rgb( ' + _r + ',' + _g + ',' + _b + ' )' );
}


sliceX = null;
sliceY = null;
sliceZ = null;

function init_volume(filename, colortable) {
  volume = new X.volume();
  volume.file = filename; //'MNI152_T1_1mm_brain.nii.gz';
  volume.labelmap.file = filename; //'MNI152_wmparc_1mm_small.nii.gz';
  volume.labelmap.colortable.file = colortable; //'FreeSurferColorLUT.txt';

  _ATLAS_ = {};
  _ATLAS_.volumes = {};
  _ATLAS_.currentMesh = 0;
  _ATLAS_.meshOpacity = 0.9;
  _ATLAS_.labelOpacity = 0.5;
  _ATLAS_.hover = null;
  _ATLAS_.hoverLabelSelect = true;

  _ATLAS_.currentVolume = volume;
  _ATLAS_.volumes[_ATLAS_.currentVolume]  = volume
};

function init_websocket(host) {
  console.info("websocket start");

  _fibers_ = {};

  $(document).ready(function () {
    _WS_ = new WebSocket(host);
    console.info("Activating stuff");
    _WS_.onopen = function () {
        console.info("websocket engage");
    };

    _WS_.onmessage = function(evt){
        console.info(evt.data)
  
        if (evt.data in _fibers_) {
          console.info("Removing tract " + evt.data);
          render3D.remove(_fibers_[evt.data]);
        };

        _fibers_[evt.data] = new X.fibers()
        _fibers_[evt.data].file = 'files/' + evt.data
        render3D.add(_fibers_[evt.data])
    };
    _WS_.onclose = function () {
        console.info("connection closed");
    };

  });
};

window.onload = function() {

  render3D = new X.renderer3D();
  render3D.init();

  render3D.onShowtime = function() {
    init_viewer2d();
  }

  render3D.add(_ATLAS_.currentVolume)
  render3D.render();

}



