import React from "https://unpkg.com/es-react@latest/dev/react.js";
import ReactDOM from "https://unpkg.com/es-react@latest/dev/react-dom.js";
import htm from "https://unpkg.com/htm@latest?module";

const html = htm.bind(React.createElement);

const FileImg = ({ file, ...props }) => {
    const [src, setSrc] = React.useState(null);
    React.useEffect(() => {
        if (file == null) {
            setSrc(null);
        }

        const url = URL.createObjectURL(file);
        setSrc(url);
        return () => {
            URL.revokeObjectURL(url);
        };
    }, [file]);

    return html`<img src=${src} ...${props} />`;
};

const App = () => {
    const [file, setFile] = React.useState(null);
    const [gradcam, setGradcam] = React.useState();
    const [predictions, setPredictions] = React.useState();

    return html`<div>
        <input
            type="file"
            disabled=${predictions === null || gradcam === null}
            onChange=${(e) => {
                setPredictions(null);
                setGradcam(undefined);
                (async () => {
                    try {
                        const file = e.target.files[0];
                        setFile(file);

                        const data = new FormData();
                        data.append("file", file);
                        setPredictions(
                            await (
                                await fetch("/predict", {
                                    method: "POST",
                                    body: data,
                                })
                            ).json()
                        );
                    } catch (e) {
                        setFile(null);
                        setGradcam(undefined);
                        setPredictions(undefined);
                    }
                })();
            }}
        />
        <p>
            ${file != null
                ? html`<div style=${{ display: "inline-block" }}>
                      <${FileImg} file=${file} height="300" />
                      <div>${"\u00A0"}</div>
                  </div>`
                : null}
            ${gradcam != null
                ? html`<div style=${{ display: "inline-block" }}>
                      <${FileImg} file=${gradcam.blob} height="300" />
                      <div>${gradcam.label}</div>
                  </div>`
                : null}
        </p>
        ${predictions != null
            ? html`<div>
                  <p>
                      prediction took
                      ${(predictions.elapsed_secs * 1000).toFixed(2)}ms.
                  </p>
                  <table>
                      <tbody>
                          ${predictions.predictions.map(
                              ({ label, score }) => html`<tr key=${label}>
                                  <td>${label}</td>
                                  <td><tt>${(score * 100).toFixed(4)}%</tt></td>
                                  <td>
                                      <button
                                          type="button"
                                          disabled=${gradcam === null}
                                          onClick=${() => {
                                              setGradcam(null);

                                              (async () => {
                                                  try {
                                                      const data =
                                                          new FormData();
                                                      data.append("file", file);
                                                      data.append(
                                                          "label",
                                                          label
                                                      );

                                                      const blob = await (
                                                          await fetch(
                                                              "/gradcam",
                                                              {
                                                                  method: "POST",
                                                                  body: data,
                                                              }
                                                          )
                                                      ).blob();
                                                      setGradcam({
                                                          label,
                                                          blob,
                                                      });
                                                  } catch (e) {
                                                      setGradcam(undefined);
                                                  }
                                              })();
                                          }}
                                      >
                                          ?
                                      </button>
                                  </td>
                              </tr>`
                          )}
                      </tbody>
                  </table>
              </div>`
            : null}
    </div>`;
};

ReactDOM.render(html`<${App} />`, document.getElementById("root"));
