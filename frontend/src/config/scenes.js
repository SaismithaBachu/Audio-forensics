export const SCENES = [
  "airport",
  "bus",
  "metro",
  "metro_station",
  "park",
  "public_square",
  "shopping_mall",
  "street_pedestrian",
  "street_traffic",
  "tram"
];

export const DEVICES = [
  {
    id: "Device A",
    label: "Device A",
    sub: "In-Ear Microphone",
    desc: "Soundman OKM II Klassik electret in-ear microphone + Zoom F8 recorder (48 kHz, 24-bit). Worn in ears, minimal head movement."
  },
  {
    id: "Device B",
    label: "Device B",
    sub: "Smartphone",
    desc: "Consumer smartphone handled in typical hand-held fashion. Time-synchronized with Device A."
  },
  {
    id: "Device C",
    label: "Device C",
    sub: "Portable Camera",
    desc: "Consumer action/compact camera handled in typical ways. Time-synchronized with Device A."
  },
];

// Helper: given a device id string like "Device A", return the full object
export const getDevice = (id) =>
  DEVICES.find((d) => d.id === id) || { label: id, sub: "", desc: "" };