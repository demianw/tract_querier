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
	//console.info(labelvalue);

	
	//volume.labelmap.opacity = 0.6;
	//volume.labelmap.showOnly = labelvalue;
        console.info(labelvalue);
        _WS_.send(labelvalue)
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

function init_websocket() {
  Error("test");
  console.info("websocket start");

  $(document).ready(function () {
  _WS_=new WebSocket("_WS_://localhost:9999/_WS_");
  console.info("Activating stuff")
  _WS_.onopen = function () {
      console.info("websocket engage");
  };

  _WS_.onmessage = function(evt){
      console.info(evt.data)
      new_fibers = new X.fibers()
      new_fibers.file = evt.data
      render3D.add(new_fibers)
      //the received content is in evt.data, set it where you want
  };

  _WS_.onclose = function () {
      console.info("connection closed");
   };

  $(".column li").on("mouseup")  //the function  that sends data
  {
      pid = $(this).attr("id");
      oldCid = $(this).parent().parent().attr("id");
      newCid = $(this).parent().attr("id");
      message = pid + " " + oldCid + " " + newCid;
      _WS_.send(message);   
  };
  });

  _WS_.send("test inside");
};

window.onload = function() {
console.info("OnLoad");
init_websocket();

render3D = new X.renderer3D();
render3D.init();

// create a mesh from a .vtk file
// // create a new X.fibers
volume = new X.volume();
volume.file = 'MNI152_T1_1mm_brain.nii.gz';
//volume.file = 't1_very_small_2.nii.gz';
//volume.color = [1, 0, 0];
//volume.labelmap.file = 'MNI152_wmparc_1mm.nii.gz';
volume.labelmap.file = 'MNI152_wmparc_1mm_small.nii.gz';
volume.labelmap.colortable.file = 'FreeSurferColorLUT.txt';
//volume.visible = false;
//volume.labelmap.visible = true;

_ATLAS_ = {};
_ATLAS_.currentVolume = volume;
_ATLAS_.volumes = {};
_ATLAS_.volumes[_ATLAS_.currentVolume]  = volume
_ATLAS_.currentMesh = 0;
_ATLAS_.meshOpacity = 0.9;
_ATLAS_.labelOpacity = 0.5;
_ATLAS_.hover = null;
_ATLAS_.hoverLabelSelect = true;

// create dictionary "label name" : "label value"
//initializeFreesurferLabels(_ATLAS_);
/*_ATLAS_.labels4 = {
  "Accumbens area": 26,
  "Amygdala": 18,
  "Caudate": 11,
  "Cerebellum Cortex": 8,
  "Cerebral Cortex": 3,
  "Hippocampus": 17,
  "Lateral Ventricle": 4,
  "Medulla": 175,
  "Midbrain": 173,
  "Pallidum": 13,
  "Pons": 174,
  "Putamen": 12,
  "Thalamus": 9,
  "Ventral DC": 28,
  "Vermis": 172,
  "3rd Ventricle": 14,
  "4th Ventricle": 15
};*/




//var fibers = new X.fibers();


//render3D.camera.position = [120, 80, 160];

render3D.add(volume);
//render3D.add(volume.labelmap);
var fibers = new X.fibers();
fibers.file = 
render3D.add(fibers);

// .. and render it
//
render3D.onShowtime = function() {
  init_viewer2d();
}
/*render3D.onShowtime = function() {

    //
    // The GUI panel
    //
    // (we need to create this during onShowtime(..) since we do not know the
    // volume dimensions before the loading was completed)
    
    // indicate if the mesh was loaded
    var gui = new dat.GUI();

    var labelmapgui = gui.addFolder('Label Map');
    var labelMapVisibleController = labelmapgui.add(volume.labelmap, 'visible');
   
    // the following configures the gui for interacting with the X.volume
    var volumegui = gui.addFolder('Volume');
    // now we can configure controllers which..
    var visibilityControler = volumegui.add(volume, 'visible')

    // .. switch between slicing and volume rendering
    //var vrController = volumegui.add(volume, 'volumeRendering');
    // the min and max color which define the linear gradient mapping
    var minColorController = volumegui.addColor(volume, 'minColor');
    var maxColorController = volumegui.addColor(volume, 'maxColor');
    // .. configure the volume rendering opacity
    var opacityController = volumegui.add(volume, 'opacity', 0, 1).listen();
    // .. and the threshold in the min..max range
    var lowerThresholdController = volumegui.add(volume, 'lowerThreshold',
        volume.min, volume.max);
    var upperThresholdController = volumegui.add(volume, 'upperThreshold',
        volume.min, volume.max);
    var lowerWindowController = volumegui.add(volume, 'windowLow', volume.min,
        volume.max);
    var upperWindowController = volumegui.add(volume, 'windowHigh', volume.min,
        volume.max);
    // the indexX,Y,Z are the currently displayed slice indices in the range
    // 0..dimensions-1
    var sliceXController = volumegui.add(volume, 'indexX', 0,
        volume.dimensions[0] - 1);
    var sliceYController = volumegui.add(volume, 'indexY', 0,
        volume.dimensions[1] - 1);
    var sliceZController = volumegui.add(volume, 'indexZ', 0,
        volume.dimensions[2] - 1);

    //var fibergui = gui.addFolder('Fiber');
    //trackVisibleController = fibergui.add(fibers, 'visible'); 
   
    //fibergui.open() 
    volumegui.open();
};*/

render3D.render();

}



